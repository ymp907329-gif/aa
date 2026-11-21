import copy

import yaml


ABLATIONS = ('baseline', 'aid', 'dfd', 'freqcl', 'full')


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def apply_ablation(config, name):
    if name not in ABLATIONS:
        raise ValueError(f'Unknown ablation: {name}. Choose from {ABLATIONS}')
    config = copy.deepcopy(config)
    model = config['model']
    loss = config['loss']
    if name == 'baseline':
        model['use_aid'] = False
        model['use_dfd'] = False
        loss['lambda_freqcl'] = 0.0
    elif name == 'aid':
        model['use_aid'] = True
        model['use_dfd'] = False
        loss['lambda_freqcl'] = 0.0
    elif name == 'dfd':
        model['use_aid'] = True
        model['use_dfd'] = True
        loss['lambda_freqcl'] = 0.0
    elif name == 'freqcl':
        model['use_aid'] = True
        model['use_dfd'] = True
        if float(loss.get('lambda_freqcl', 0.0)) <= 0:
            loss['lambda_freqcl'] = 0.1
    return config


def create_model(config):
    from models.decompfuse import DecompFuse

    model_config = config['model']
    model = DecompFuse(
        in_ch=model_config.get('in_ch', 1),
        base_ch=model_config.get('base_ch', 64),
        use_aid=model_config.get('use_aid', True),
        use_dfd=model_config.get('use_dfd', True),
        use_skip=model_config.get('use_skip', False),
        use_adaptive_intensity=model_config.get('use_adaptive_intensity', False),
        use_source_reconstruction=model_config.get('use_source_reconstruction', False),
        source_reconstruction_mode=model_config.get('source_reconstruction_mode', 'features'),
        aid_high_only=model_config.get('aid_high_only', False),
        use_global=model_config.get('use_global', False),
        global_pool=model_config.get('global_pool', 8),
        global_heads=model_config.get('global_heads', 4),
        output_anchor=model_config.get('output_anchor', False),
        anchor_residual_scale=model_config.get('anchor_residual_scale', 0.1),
        use_adaptive_compensation=model_config.get('use_adaptive_compensation', False),
        compensation_config=model_config.get('compensation'),
        aid_config=model_config.get('aid'),
        dfd_config=model_config.get('dfd'),
    )
    normalization = model_config.get('normalization', 'batch')
    if normalization == 'group':
        import math
        import torch.nn as nn

        for name, module in list(model.named_modules()):
            if isinstance(module, nn.BatchNorm2d):
                parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
                groups = math.gcd(8, module.num_features)
                setattr(parent, name.rsplit('.', 1)[-1], nn.GroupNorm(groups, module.num_features))
    elif normalization != 'batch':
        raise ValueError("model.normalization must be 'batch' or 'group'")
    return model
