import time

import torch


def _conv_flops(module, inputs, output):
    batch = output.shape[0]
    out_channels = output.shape[1]
    output_pixels = output.shape[2] * output.shape[3]
    kernel_ops = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
    module._fusion_flops = getattr(module, '_fusion_flops', 0) + (
        2 * batch * out_channels * output_pixels * kernel_ops
    )


def _transpose_conv_flops(module, inputs, output):
    batch, in_channels, height, width = inputs[0].shape
    kernel_ops = module.out_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
    module._fusion_flops = getattr(module, '_fusion_flops', 0) + (
        2 * batch * in_channels * height * width * kernel_ops
    )


def measure_efficiency(model, ir, vi, warmup=3, repeats=10):
    was_training = model.training
    model.eval()
    hooks = []
    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(_conv_flops))
            modules.append(module)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            hooks.append(module.register_forward_hook(_transpose_conv_flops))
            modules.append(module)
    try:
        with torch.no_grad():
            for module in modules:
                module._fusion_flops = 0
            model(ir, vi)
            flops = sum(getattr(module, '_fusion_flops', 0) for module in modules)
            for _ in range(warmup):
                model(ir, vi)
            if ir.device.type == 'cuda':
                torch.cuda.synchronize(ir.device)
            start = time.perf_counter()
            for _ in range(repeats):
                model(ir, vi)
            if ir.device.type == 'cuda':
                torch.cuda.synchronize(ir.device)
            elapsed = time.perf_counter() - start
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()
    latency = elapsed / repeats * 1000
    return {
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'flops': int(flops),
        'gflops': flops / 1e9,
        'latency_ms': latency,
        'fps': 1000 / latency,
    }
