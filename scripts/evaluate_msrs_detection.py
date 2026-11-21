import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / 'external' / 'yolov5'
sys.path.insert(0, str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ap_per_class, process_batch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--data_root', default='data/MSRS/downstream')
    parser.add_argument('--weights', default='external/yolov5/yolov5s.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    root = Path(args.data_root) / args.method
    config = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))
    model = DetectMultiBackend(str(PROJECT_ROOT / args.weights), device=device, fp16=False)
    model.eval()
    loader = create_dataloader(
        str(root / 'images'), args.imgsz, args.batch_size, int(model.stride),
        pad=0.5, rect=False, workers=0, prefix='val: ',
    )[0]
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []
    for images, targets, paths, shapes in loader:
        images = images.to(device).float() / 255
        targets = targets.to(device)
        predictions = model(images)
        height, width = images.shape[-2:]
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        predictions = non_max_suppression(
            predictions, conf_thres=0.001, iou_thres=0.6, multi_label=True, max_det=300
        )
        for index, prediction in enumerate(predictions):
            labels = targets[targets[:, 0] == index, 1:]
            prediction = prediction[prediction[:, 5] < 3]
            correct = torch.zeros(prediction.shape[0], 10, dtype=torch.bool, device=device)
            if prediction.shape[0] and labels.shape[0]:
                prediction_boxes = prediction.clone()
                scale_boxes(images[index].shape[1:], prediction_boxes[:, :4], shapes[index][0], shapes[index][1])
                target_boxes = xywh2xyxy(labels[:, 1:5])
                scale_boxes(images[index].shape[1:], target_boxes, shapes[index][0], shapes[index][1])
                target_labels = torch.cat((labels[:, 0:1], target_boxes), dim=1)
                correct = process_batch(prediction_boxes, target_labels, iouv)
            stats.append((correct, prediction[:, 4], prediction[:, 5], labels[:, 0]))
    if not stats:
        raise ValueError('No detection samples evaluated')
    arrays = [torch.cat(items, 0).cpu().numpy() for items in zip(*stats)]
    names = {0: 'person', 1: 'bicycle', 2: 'car'}
    tp, fp, precision, recall, f1, ap, ap_class = ap_per_class(
        *arrays, plot=False, save_dir=root, names=names
    )
    target_count = np.bincount(arrays[3].astype(int), minlength=3)
    ap50 = ap[:, 0]
    ap_all = ap.mean(axis=1)
    result = {
        'method': args.method,
        'weights': args.weights,
        'image_count': len(loader.dataset),
        'instance_count': int(target_count.sum()),
        'precision': float(precision.mean()),
        'recall': float(recall.mean()),
        'map50': float(ap50.mean()),
        'map50_95': float(ap_all.mean()),
        'per_class': {
            names[int(class_id)]: {
                'instances': int(target_count[int(class_id)]),
                'precision': float(precision[index]),
                'recall': float(recall[index]),
                'ap50': float(ap50[index]),
                'ap50_95': float(ap_all[index]),
            }
            for index, class_id in enumerate(ap_class)
        },
        'protocol': 'Fixed pretrained YOLOv5s; 256x256 images resized by the detector to 640; classes person/bicycle/car.',
    }
    output = PROJECT_ROOT / 'results' / 'downstream' / 'msrs' / args.method
    output.mkdir(parents=True, exist_ok=True)
    with open(output / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
