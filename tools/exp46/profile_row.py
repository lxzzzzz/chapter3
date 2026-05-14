import argparse
import json
import time
from pathlib import Path

from common import EXP_DIR, PROFILE_VARIANTS, ROOT_DIR, write_cfg, write_metrics


def main():
    parser = argparse.ArgumentParser(description='Profile one Chapter 4.6 complexity row.')
    parser.add_argument('--row', required=True, choices=list(PROFILE_VARIANTS.keys()))
    parser.add_argument('--dataset', default='mths', choices=['mths', 'dair', 'v2x_real'])
    args = parser.parse_args()

    variant = PROFILE_VARIANTS[args.row]
    cfg_path = write_cfg(variant, args.dataset, 1, f'4-21_{args.row}_{args.dataset}')
    payload = {
        'table': '4-21',
        'row': args.row,
        'variant': variant,
        'dataset': args.dataset,
        'config': str(cfg_path),
    }
    start = time.time()
    try:
        import sys
        sys.path.insert(0, str(ROOT_DIR / 'tools'))
        import _init_path  # noqa: F401
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network
        from pcdet.utils import common_utils

        cfg_from_yaml_file(str(cfg_path), cfg)
        logger = common_utils.create_logger()
        dataset, _, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=0,
            logger=logger,
            training=False,
        )
        model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
        payload['params'] = sum(p.numel() for p in model.parameters())
        payload['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception as exc:
        payload['profile_error'] = repr(exc)
    payload['elapsed_sec'] = time.time() - start
    path = write_metrics('4-21', args.row, payload)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
