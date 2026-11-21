from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def load_image(path, size):
    with Image.open(path) as image:
        return np.asarray(
            image.convert('L').resize((size, size), Image.BILINEAR), dtype=np.float64
        ) / 255


def image_features(array):
    histogram = np.histogram(array, bins=32, range=(0, 1))[0]
    probability = histogram / histogram.sum()
    entropy = -np.sum(probability[probability > 0] * np.log(probability[probability > 0]))
    gradient = np.abs(np.diff(array, axis=0)).mean() + np.abs(np.diff(array, axis=1)).mean()
    return np.array([array.mean(), array.std(), entropy, gradient])


def source_array_features(ir_image, vi_image):
    return np.concatenate((
        image_features(ir_image), image_features(vi_image),
        [np.abs(ir_image - vi_image).mean(), np.corrcoef(ir_image.ravel(), vi_image.ravel())[0, 1]],
    ))


def source_features(data_root, split, names, size):
    features = []
    for name in names:
        images = []
        for modality in ('ir', 'vi'):
            candidates = [
                path for path in (Path(data_root) / split / modality).glob(f'{Path(name).stem}.*')
                if path.suffix.lower() in VALID_EXTENSIONS
            ]
            if len(candidates) != 1:
                raise ValueError(f'Expected one {modality} image for {name}, found {len(candidates)}')
            images.append(load_image(candidates[0], size))
        features.append(source_array_features(*images))
    return np.asarray(features)


def ridge_fit(train_x, train_y):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0) + 1e-8
    normalized = (train_x - mean) / scale
    target_mean = train_y.mean()
    weights = np.linalg.solve(
        normalized.T @ normalized + np.eye(train_x.shape[1]),
        normalized.T @ (train_y - target_mean),
    )
    return {'weights': weights, 'intercept': target_mean, 'feature_mean': mean, 'feature_scale': scale}


def ridge_apply(model, test_x):
    normalized = (test_x - model['feature_mean']) / model['feature_scale']
    return model['intercept'] + normalized @ model['weights']


class ValueDepthController:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.data = {key: data[key] for key in data.files}
        self.target = str(self.data['target'])
        self.depths = self.data['depths'].astype(int)
        self.size = int(self.data['size'])

    def select(self, ir, vi):
        features = source_array_features(ir, vi)[None, :]
        if self.target != 'balanced':
            model = {
                'weights': self.data['weights'], 'intercept': self.data['intercept'],
                'feature_mean': self.data['feature_mean'], 'feature_scale': self.data['feature_scale'],
            }
            return 4 if float(ridge_apply(model, features)[0]) > 0 else 3
        predictions = np.zeros((len(self.depths), len(self.data['metric_mean'])))
        for depth_index in range(len(self.depths)):
            for metric_index in range(predictions.shape[1]):
                model = {
                    'weights': self.data[f'weights_{depth_index}_{metric_index}'],
                    'intercept': self.data[f'intercept_{depth_index}_{metric_index}'],
                    'feature_mean': self.data['feature_mean'],
                    'feature_scale': self.data['feature_scale'],
                }
                predictions[depth_index, metric_index] = ridge_apply(model, features)[0]
        utility = ((predictions - self.data['metric_mean']) / self.data['metric_scale']).mean(axis=1)
        return int(self.depths[int(utility.argmax())])
