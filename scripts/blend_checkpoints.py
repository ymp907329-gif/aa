import argparse
import os

import torch

from train import make_loaders, validate, validation_score
from utils.config import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--finetuned', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_val_batches', type=int)
    args = parser.parse_args()
    if any(not 0 <= alpha <= 1 for alpha in args.alphas):
        raise ValueError('Blend alphas must be in [0, 1]')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    base = torch.load(args.base, map_location=device)
    finetuned = torch.load(args.finetuned, map_location=device)
    cfg = finetuned['config']
    _, val_loader = make_loaders(cfg)
    model = create_model(cfg).to(device)
    reference = cfg['train'].get('mi_qabf_reference')
    best = None
    for alpha in args.alphas:
        state = {}
        for key, value in finetuned['model'].items():
            if key not in base['model']:
                state[key] = value
            elif value.is_floating_point():
                state[key] = (1 - alpha) * base['model'][key] + alpha * value
            else:
                state[key] = value
        model.load_state_dict(state)
        values = validate(
            model, val_loader, device, args.max_val_batches, selection_metric='mi_qabf'
        )
        score = validation_score(values, 'mi_qabf', reference)
        print(f'alpha={alpha:.2f}, score={score:.6f}, {values}')
        if best is None or score > best[0]:
            best = score, alpha, state, values

    score, alpha, state, values = best
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model': state,
        'config': cfg,
        'epoch': 0,
        'best_score': score,
        'best_psnr': values['psnr'],
        'selection_metric': 'mi_qabf',
        'blend': {'base': args.base, 'finetuned': args.finetuned, 'alpha': alpha},
    }, args.output)
    print(f'Saved best blend: alpha={alpha:.2f}, score={score:.6f}, output={args.output}')


if __name__ == '__main__':
    main()
