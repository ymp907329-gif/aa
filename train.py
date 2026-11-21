import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import FusionDataset
from data.transforms import Compose, JointGaussianNoise, JointPhotometric, JointRandomFlip
from losses.fusion_loss import FusionLoss, teacher_gradient_distillation
from utils.checkpoint import save_checkpoint
from utils.config import ABLATIONS, apply_ablation, create_model, load_config
from utils.logger import Logger
from utils.metrics import compute_mi, compute_psnr, compute_qabf, compute_sf, compute_ssim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_conflicting_gradients(task_losses, parameters):
    parameters = list(parameters)
    task_gradients = []
    for loss in task_losses.values():
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        task_gradients.append(list(gradients))
    adjusted = [list(gradients) for gradients in task_gradients]
    random_state = random.getstate()
    for index in range(len(adjusted)):
        order = list(range(len(adjusted)))
        random.shuffle(order)
        for other in order:
            if index == other:
                continue
            dot = None
            norm_squared = None
            for current, reference in zip(adjusted[index], task_gradients[other]):
                if current is None or reference is None:
                    continue
                dot_value = (current * reference).sum()
                norm_value = reference.square().sum()
                dot = dot_value if dot is None else dot + dot_value
                norm_squared = norm_value if norm_squared is None else norm_squared + norm_value
            if dot is not None and dot.item() < 0:
                for position, (current, reference) in enumerate(zip(
                    adjusted[index], task_gradients[other]
                )):
                    if current is not None and reference is not None:
                        adjusted[index][position] = current - dot * reference / (
                            norm_squared + 1e-12
                        )
    random.setstate(random_state)
    combined = []
    for position in range(len(parameters)):
        gradients = [gradients[position] for gradients in adjusted if gradients[position] is not None]
        combined.append(torch.stack(gradients).sum(dim=0) if gradients else None)
    return combined


def make_loaders(cfg):
    data = cfg['data']
    exclude_names = None
    if data.get('exclude_test_overlap', False):
        test_ds = FusionDataset(data['root'], split=data['test_split'], size=data['size'])
        exclude_names = set(test_ds.names)
    augmentation = data.get('augmentation', {})
    transforms = [JointRandomFlip(p=augmentation.get('flip_probability', 0.5))]
    if augmentation.get('photometric', False):
        transforms.append(JointPhotometric(
            probability=augmentation.get('probability', 0.8),
            brightness=augmentation.get('brightness', 0.08),
            contrast=augmentation.get('contrast', 0.2),
            gamma=augmentation.get('gamma', 0.2),
        ))
    if augmentation.get('gaussian_noise', False):
        transforms.append(JointGaussianNoise(
            probability=augmentation.get('noise_probability', 0.35),
            sigma_min=augmentation.get('noise_sigma_min', 0.01),
            sigma_max=augmentation.get('noise_sigma_max', 0.03),
        ))
    train_ds = FusionDataset(
        data['root'], split=data['train_split'], size=data['size'],
        transform=Compose(transforms), exclude_names=exclude_names,
    )
    if exclude_names:
        all_train = FusionDataset(data['root'], split=data['train_split'], size=data['size'])
        print(f'Excluded {len(all_train) - len(train_ds)} training samples overlapping the test split')
    if data.get('val_split'):
        val_ds = FusionDataset(
            data['root'], split=data['val_split'], size=data['size'], exclude_names=exclude_names
        )
    else:
        val_count = max(1, round(len(train_ds) * data.get('val_ratio', 0.1)))
        if val_count >= len(train_ds):
            raise ValueError('Validation split leaves no training samples')
        split_seed = data.get('split_seed', cfg['train']['seed'])
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(split_seed))
        val_indices = indices[:val_count].tolist()
        train_indices = indices[val_count:].tolist()
        val_ds = Subset(FusionDataset(
            data['root'], split=data['train_split'], size=data['size'], exclude_names=exclude_names
        ), val_indices)
        train_ds = Subset(train_ds, train_indices)
    common = {'num_workers': data.get('num_workers', 0), 'pin_memory': torch.cuda.is_available()}
    train_loader = DataLoader(
        train_ds, batch_size=data['batch_size'], shuffle=True, drop_last=False, **common
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **common)
    return train_loader, val_loader


def train_one_epoch(
    model, loader, optimizer, criterion, device, cfg, logger, epoch, max_batches=None, teacher=None
):
    model.train()
    gradient_surgery = cfg['train'].get('gradient_surgery', False)
    if gradient_surgery and (teacher is not None or cfg['train'].get('multi_depth', False)):
        raise ValueError('Gradient surgery currently requires single-depth training without distillation')
    totals = {'loss': 0.0, 'aid_steps': 0.0}
    count = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for step, batch in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        task_losses = None
        if cfg['train'].get('multi_depth', False):
            candidates = model.forward_iterations(ir, vi)
            weights = cfg['train'].get('depth_loss_weights', [1.0] * len(candidates))
            if len(weights) != len(candidates):
                raise ValueError('depth_loss_weights must match the number of AID iterations')
            loss = ir.new_zeros(())
            for weight, (candidate, candidate_aux) in zip(weights, candidates):
                candidate_loss, components = criterion(candidate, ir, vi, candidate_aux)
                loss = loss + weight * candidate_loss
            out, aux = candidates[-1]
        else:
            out, aux = model(ir, vi, return_aux=True)
            if gradient_surgery:
                loss, components, task_losses = criterion(
                    out, ir, vi, aux, return_task_losses=True
                )
            else:
                loss, components = criterion(out, ir, vi, aux)
        if teacher is not None:
            with torch.no_grad():
                teacher_out = teacher(ir, vi)
            distillation = F.l1_loss(out, teacher_out)
            loss = loss + cfg['train'].get('lambda_distill', 0.1) * distillation
            components['distill'] = distillation.detach()
            teacher_gradient_weight = cfg['train'].get('lambda_teacher_gradient', 0.0)
            if teacher_gradient_weight > 0:
                teacher_gradient = teacher_gradient_distillation(
                    out, teacher_out, ir, vi,
                    max_ratio=cfg['train'].get('teacher_gradient_max_ratio', 1.5),
                    min_cosine=cfg['train'].get('teacher_gradient_min_cosine', 0.0),
                )
                loss = loss + teacher_gradient_weight * teacher_gradient
                components['teacher_gradient'] = teacher_gradient.detach()
        optimizer.zero_grad()
        decoder_gradients = None
        decoder_parameters = None
        if gradient_surgery:
            primary_tasks = {
                name: task for name, task in task_losses.items()
                if name in ('pixel', 'gradient', 'ssim') and task.detach().abs().item() > 0
            }
            other_tasks = [
                task for name, task in task_losses.items()
                if name not in ('pixel', 'gradient', 'ssim') and task.detach().abs().item() > 0
            ]
            decoder_parameters = list(model.decoder.parameters())
            decoder_gradients = project_conflicting_gradients(primary_tasks, decoder_parameters)
            if other_tasks:
                other_loss = torch.stack(other_tasks).sum()
                other_gradients = torch.autograd.grad(
                    other_loss, decoder_parameters, retain_graph=True, allow_unused=True
                )
                decoder_gradients = [
                    primary if other is None else primary + other
                    if primary is not None else other
                    for primary, other in zip(decoder_gradients, other_gradients)
                ]
        loss.backward()
        if gradient_surgery:
            for parameter, gradient in zip(decoder_parameters, decoder_gradients):
                if gradient is not None:
                    parameter.grad = gradient
        optimizer.step()

        count += 1
        totals['loss'] += loss.item()
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + value.item()
        if 'aid' in aux:
            totals['aid_steps'] += aux['aid']['mean_steps'].item()
            for key in ('consistency', 'stop_score'):
                if key in aux['aid']:
                    totals[key] = totals.get(key, 0.0) + aux['aid'][key].item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        if step % cfg['train']['log_freq'] == 0:
            global_step = (epoch - 1) * len(loader) + step
            logger.scalar('train/loss', loss.item(), global_step)
            for key, value in components.items():
                logger.scalar(f'train/{key}', value.item(), global_step)
            if 'aid' in aux:
                logger.scalar('train/aid_steps', aux['aid']['mean_steps'].item(), global_step)
                for key in ('consistency', 'stop_score'):
                    if key in aux['aid']:
                        logger.scalar(f'train/aid_{key}', aux['aid'][key].item(), global_step)
    if count == 0:
        raise ValueError('No training batches were processed')
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device, max_batches=None, selection_metric='ssim'):
    model.eval()
    psnr_list, ssim_list = [], []
    mi_list, qabf_list, sf_list = [], [], []
    needs_mi = selection_metric in ('mi', 'mi_qabf')
    needs_qabf = selection_metric in ('qabf', 'mi_qabf', 'sf_qabf')
    needs_sf = selection_metric in ('sf', 'sf_qabf')
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        out = model(ir, vi)
        for i in range(out.shape[0]):
            fused = out[i].cpu().numpy()
            ir_image = ir[i].cpu().numpy()
            vi_image = vi[i].cpu().numpy()
            psnr_list.append(0.5 * (
                compute_psnr(fused, ir_image) + compute_psnr(fused, vi_image)
            ))
            ssim_list.append(0.5 * (
                compute_ssim(fused, ir_image) + compute_ssim(fused, vi_image)
            ))
            if needs_mi:
                mi_list.append(compute_mi(ir_image, fused) + compute_mi(vi_image, fused))
            if needs_qabf:
                qabf_list.append(compute_qabf(ir_image, vi_image, fused))
            if needs_sf:
                sf_list.append(compute_sf(fused))
    if not psnr_list:
        raise ValueError('No validation samples were evaluated')
    values = {
        'psnr': float(np.mean(psnr_list)),
        'ssim': float(np.mean(ssim_list)),
    }
    if needs_mi:
        values['mi'] = float(np.mean(mi_list))
    if needs_qabf:
        values['qabf'] = float(np.mean(qabf_list))
    if needs_sf:
        values['sf'] = float(np.mean(sf_list))
    return values


def validation_score(values, selection_metric, reference=None):
    if selection_metric in ('psnr', 'ssim', 'mi', 'qabf', 'sf'):
        return values[selection_metric]
    if selection_metric in ('mi_qabf', 'sf_qabf'):
        first, second = selection_metric.split('_')
        if not reference or reference.get(first, 0) <= 0 or reference.get(second, 0) <= 0:
            raise ValueError(f'{selection_metric} selection requires positive validation references')
        return 0.5 * (
            values[first] / reference[first] + values[second] / reference[second]
        )
    raise ValueError("selection_metric must be 'psnr', 'ssim', 'mi', 'qabf', 'sf', 'mi_qabf', or 'sf_qabf'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ablation', choices=ABLATIONS, default='full')
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--resume')
    checkpoint_group.add_argument('--init_from')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_dir')
    parser.add_argument('--log_dir')
    parser.add_argument('--distill_from')
    parser.add_argument('--max_train_batches', type=int)
    parser.add_argument('--max_val_batches', type=int)
    args = parser.parse_args()

    cfg = apply_ablation(load_config(args.config), args.ablation)
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.seed is not None:
        cfg['train']['seed'] = args.seed
    if args.distill_from:
        cfg['train']['distill_from'] = args.distill_from
    if args.save_dir:
        cfg['train']['save_dir'] = args.save_dir
    if args.log_dir:
        cfg['train']['log_dir'] = args.log_dir
    set_seed(cfg['train']['seed'])
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}; ablation: {args.ablation}')
    train_loader, val_loader = make_loaders(cfg)
    print(f'Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}')

    if not cfg['data'].get('exclude_test_overlap', False):
        test_ir = os.path.join(cfg['data']['root'], cfg['data']['test_split'], 'ir')
        train_ir = os.path.join(cfg['data']['root'], cfg['data']['train_split'], 'ir')
        if os.path.isdir(test_ir):
            overlap = set(os.listdir(train_ir)) & set(os.listdir(test_ir))
            if overlap:
                print(f'WARNING: train/test share {len(overlap)} filenames; test metrics are not independent.')

    model = create_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    criterion = FusionLoss(cfg['loss']).to(device)
    logger = Logger(cfg['train']['log_dir'])
    selection_metric = cfg['train'].get('selection_metric', 'psnr')
    if selection_metric not in ('psnr', 'ssim', 'mi', 'qabf', 'sf', 'mi_qabf', 'sf_qabf'):
        raise ValueError("selection_metric must be 'psnr', 'ssim', 'mi', 'qabf', 'sf', 'mi_qabf', or 'sf_qabf'")
    selection_reference = cfg['train'].get(f'{selection_metric}_reference')
    if selection_metric in ('mi_qabf', 'sf_qabf') and (
        not selection_reference
        or any(selection_reference.get(key, 0) <= 0 for key in selection_metric.split('_'))
    ):
        raise ValueError(f'{selection_metric} selection requires positive validation references')
    start_epoch = 1
    best_score = float('-inf')
    best_psnr = float('-inf')
    no_improve = 0
    init_from = None if args.resume else (args.init_from or cfg['train'].get('init_from'))
    if init_from:
        ckpt = torch.load(init_from, map_location=device)
        if cfg['train'].get('init_strict', True):
            model.load_state_dict(ckpt['model'])
        else:
            current = model.state_dict()
            compatible = {
                key: value for key, value in ckpt['model'].items()
                if key in current and current[key].shape == value.shape
            }
            model.load_state_dict(compatible, strict=False)
            print(f'Loaded {len(compatible)} compatible tensors from {init_from}')
        print(f'Initialized weights from {init_from}, epoch {ckpt.get("epoch", "?")}')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', best_psnr)
        best_score = ckpt.get('best_score', best_psnr)
        print(f'Resumed from {args.resume}, epoch {start_epoch}')

    teacher = None
    distill_from = cfg['train'].get('distill_from')
    if distill_from:
        teacher_type = cfg['train'].get('teacher_type', 'decompfuse')
        if teacher_type == 'cddfuse':
            from models.teachers import CDDFuseTeacher
            teacher = CDDFuseTeacher(distill_from).to(device)
        elif teacher_type == 'decompfuse':
            teacher_ckpt = torch.load(distill_from, map_location=device)
            teacher = create_model(teacher_ckpt.get('config', cfg)).to(device)
            teacher.load_state_dict(teacher_ckpt['model'])
        else:
            raise ValueError(f'Unknown teacher type: {teacher_type}')
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f'Distillation teacher: {teacher_type}, {distill_from}')

    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        train_values = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, logger, epoch,
            args.max_train_batches, teacher=teacher,
        )
        scheduler.step()
        val_values = validate(
            model, val_loader, device, args.max_val_batches, selection_metric=selection_metric
        )
        psnr, ssim = val_values['psnr'], val_values['ssim']
        score = validation_score(val_values, selection_metric, selection_reference)
        improved = score > best_score
        if improved:
            best_score = score
            best_psnr = psnr
            no_improve = 0
        else:
            no_improve += 1
        message = (
            f'[Epoch {epoch}] loss={train_values["loss"]:.4f}, '
            f'PSNR={psnr:.2f}, SSIM={ssim:.4f}, AID steps={train_values["aid_steps"]:.2f}'
        )
        if 'mi' in val_values:
            message += f', MI={val_values["mi"]:.4f}'
        if 'qabf' in val_values:
            message += f', Qabf={val_values["qabf"]:.4f}'
        if 'sf' in val_values:
            message += f', SF={val_values["sf"]:.4f}'
        print(message)
        for key, value in val_values.items():
            logger.scalar(f'val/{key}', value, epoch)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_score': best_score,
            'best_psnr': best_psnr,
            'selection_metric': selection_metric,
            'config': cfg,
        }
        save_checkpoint(state, cfg['train']['save_dir'], 'last.pth')
        if improved:
            save_checkpoint(state, cfg['train']['save_dir'], 'best.pth')
        if epoch % cfg['train']['save_freq'] == 0:
            save_checkpoint(state, cfg['train']['save_dir'], f'epoch_{epoch}.pth')
        patience = cfg['train'].get('patience', 0)
        if patience and no_improve >= patience:
            print(f'Early stop at epoch {epoch}. Best {selection_metric}: {best_score:.4f}')
            break
    logger.close()
    print(f'Training finished. Best {selection_metric}: {best_score:.4f}, PSNR: {best_psnr:.2f}')


if __name__ == '__main__':
    main()
