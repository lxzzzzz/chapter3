import argparse

from common import ROOT_DIR, write_cfg


def main():
    parser = argparse.ArgumentParser(description='Smoke-test a Chapter 4.6 model forward pass.')
    parser.add_argument('--dataset', default='mths', choices=['mths', 'dair', 'v2x_real'])
    parser.add_argument('--variant', default='full')
    parser.add_argument('--num-batches', type=int, default=2)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT_DIR / 'tools'))
    import _init_path  # noqa: F401
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
    from pcdet.utils import common_utils

    cfg_path = write_cfg(args.variant, args.dataset, 1, f'smoke_{args.variant}_{args.dataset}')
    cfg_from_yaml_file(str(cfg_path), cfg)
    logger = common_utils.create_logger()
    _, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=True,
    )
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), loader.dataset).cuda().train()
    model_fn = model_fn_decorator()
    for idx, batch_dict in enumerate(loader):
        if idx >= args.num_batches:
            break
        load_data_to_gpu(batch_dict)
        ret, tb_dict, _ = model_fn(model, batch_dict)
        loss = ret['loss'] if isinstance(ret, dict) else ret
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at batch {idx}: {loss}')
        print(f'batch={idx} loss={float(loss):.6f} tb_keys={sorted(tb_dict.keys())[:8]}')


if __name__ == '__main__':
    main()
