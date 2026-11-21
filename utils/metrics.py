import numpy as np
from scipy.ndimage import gaussian_filter
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _squeeze_image(img):
    img = _to_numpy(img)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    return np.asarray(img, dtype=np.float64)


def compute_psnr(img1, img2, data_range=1.0):
    img1 = _squeeze_image(img1)
    img2 = _squeeze_image(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(img1, img2, data_range=1.0):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        _squeeze_image(img1), _squeeze_image(img2), data_range=data_range
    ))


def compute_entropy(img, bins=256):
    img = _squeeze_image(img)
    hist, _ = np.histogram(img, bins=bins, range=(0, 1))
    probabilities = hist.astype(np.float64) / max(hist.sum(), 1)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


PIXEL_SCALE = 255.0


def compute_sd(img):
    return float(np.std(_squeeze_image(img)) * PIXEL_SCALE)


def compute_ag(img):
    img = _squeeze_image(img)
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    dx = dx[:dy.shape[0], :]
    dy = dy[:, :dx.shape[1]]
    return float(np.sqrt((dx ** 2 + dy ** 2) / 2).mean() * PIXEL_SCALE)


def compute_sf(img):
    img = _squeeze_image(img)
    rf = np.diff(img, axis=0)
    cf = np.diff(img, axis=1)
    return float(np.sqrt(np.mean(rf ** 2) + np.mean(cf ** 2)) * PIXEL_SCALE)


def compute_vif(reference, fused, sigma_nsq=2.0, scales=4):
    reference = _squeeze_image(reference)
    fused = _squeeze_image(fused)
    numerator = 0.0
    denominator = 0.0
    for scale in range(scales):
        if min(reference.shape) < 16:
            break
        if scale:
            reference = reference[::2, ::2]
            fused = fused[::2, ::2]
        mu_ref = gaussian_filter(reference, sigma=2.0)
        mu_fused = gaussian_filter(fused, sigma=2.0)
        var_ref = np.maximum(gaussian_filter(reference * reference, sigma=2.0) - mu_ref ** 2, 0)
        var_fused = np.maximum(gaussian_filter(fused * fused, sigma=2.0) - mu_fused ** 2, 0)
        covariance = gaussian_filter(reference * fused, sigma=2.0) - mu_ref * mu_fused
        gain = covariance / (var_ref + 1e-10)
        noise_var = np.maximum(var_fused - gain * covariance, 0)
        valid = var_ref > 1e-10
        information = np.log2(1 + (gain ** 2 * var_ref) / (noise_var + sigma_nsq))
        total_information = np.log2(1 + var_ref / sigma_nsq)
        numerator += float(information[valid].sum())
        denominator += float(total_information[valid].sum())
    return numerator / (denominator + 1e-12)


def compute_mi(img1, img2, bins=256):
    img1 = _squeeze_image(img1)
    img2 = _squeeze_image(img2)
    joint, _, _ = np.histogram2d(
        img1.ravel(), img2.ravel(), bins=bins, range=((0, 1), (0, 1))
    )
    joint = joint.astype(np.float64)
    joint /= max(joint.sum(), 1)
    marginal_1 = joint.sum(axis=1, keepdims=True)
    marginal_2 = joint.sum(axis=0, keepdims=True)
    independent = marginal_1 * marginal_2
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log2(joint[valid] / independent[valid])))


def _edge_map(img):
    from scipy.ndimage import convolve

    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gx = convolve(img, kernel_x, mode='nearest')
    gy = convolve(img, kernel_y, mode='nearest')
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def _edge_preservation(source_strength, source_angle, fused_strength, fused_angle):
    min_strength = np.minimum(source_strength, fused_strength)
    max_strength = np.maximum(source_strength, fused_strength)
    strength_ratio = np.where(max_strength > 0, min_strength / (max_strength + 1e-12), 1.0)
    angle_delta = np.abs(source_angle - fused_angle) % np.pi
    angle_delta = np.minimum(angle_delta, np.pi - angle_delta)
    angle_ratio = 1 - angle_delta / (np.pi / 2)
    strength_score = 0.9994 / (1 + np.exp(-15 * (strength_ratio - 0.5)))
    angle_score = 0.9879 / (1 + np.exp(-22 * (angle_ratio - 0.8)))
    return strength_score * angle_score


def compute_qabf(ir, vi, fused):
    ir = _squeeze_image(ir)
    vi = _squeeze_image(vi)
    fused = _squeeze_image(fused)
    ir_strength, ir_angle = _edge_map(ir)
    vi_strength, vi_angle = _edge_map(vi)
    fused_strength, fused_angle = _edge_map(fused)
    ir_score = _edge_preservation(ir_strength, ir_angle, fused_strength, fused_angle)
    vi_score = _edge_preservation(vi_strength, vi_angle, fused_strength, fused_angle)
    numerator = np.sum(ir_score * ir_strength + vi_score * vi_strength)
    denominator = np.sum(ir_strength + vi_strength)
    return float(numerator / (denominator + 1e-12))


def compute_fusion_metrics(fused, ir, vi):
    return {
        'psnr': 0.5 * (compute_psnr(fused, ir) + compute_psnr(fused, vi)),
        'ssim': 0.5 * (compute_ssim(fused, ir) + compute_ssim(fused, vi)),
        'vif': 0.5 * (compute_vif(ir, fused) + compute_vif(vi, fused)),
        'entropy': compute_entropy(fused),
        'sd': compute_sd(fused),
        'ag': compute_ag(fused),
        'sf': compute_sf(fused),
        'mi': compute_mi(ir, fused) + compute_mi(vi, fused),
        'qabf': compute_qabf(ir, vi, fused),
    }
