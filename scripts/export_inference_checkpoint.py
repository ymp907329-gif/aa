import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location='cpu')
    state = checkpoint['model']
    removed = [key for key in state if key.startswith(('source_decoder_ir.', 'source_decoder_vi.'))]
    for key in removed:
        del state[key]
    config = checkpoint.get('config')
    if config and 'model' in config:
        config['model']['use_source_reconstruction'] = False
    checkpoint.pop('optimizer', None)
    checkpoint.pop('scheduler', None)
    checkpoint['export'] = {'removed_training_only_tensors': len(removed)}
    torch.save(checkpoint, args.output)
    print(f'Removed {len(removed)} training-only tensors; saved {args.output}')


if __name__ == '__main__':
    main()
