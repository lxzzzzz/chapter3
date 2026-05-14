import argparse
import os
from pathlib import Path

from common import ROOT_DIR, dataset_specs


def ensure_link(link_path, target_path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return
    os.symlink(target_path, link_path)


def prepare_overlay(spec):
    local = ROOT_DIR / spec.data_path
    external = Path(spec.external_path)
    if spec.key == 'dair':
        ensure_link(local, external)
        return
    local.mkdir(parents=True, exist_ok=True)
    for name in ('ImageSets', 'training'):
        ensure_link(local / name, external / name)


def main():
    parser = argparse.ArgumentParser(description='Check Chapter 4.6 dataset roots and info files.')
    parser.add_argument('--no-link', action='store_true', help='only check paths, do not create local symlink overlays')
    args = parser.parse_args()

    ok = True
    for spec in dataset_specs().values():
        if not args.no_link:
            prepare_overlay(spec)
        local = ROOT_DIR / spec.data_path
        external = Path(spec.external_path)
        print(f'[{spec.key}] external={external}')
        print(f'[{spec.key}] local={local}')
        if not external.exists():
            print(f'  MISSING external root')
            ok = False
        for split, names in spec.info_path.items():
            for name in names:
                path = local / name
                exists = path.exists()
                print(f'  {split}: {path} {"OK" if exists else "MISSING"}')
                ok = ok and exists
        if spec.key == 'mths':
            print('  build if missing: python tools/create_v2x_xian_infos.py --data_path data/v2x_xian --save_path data/v2x_xian --splits train val')
        elif spec.key == 'v2x_real':
            print('  build if missing: python tools/create_v2x_real_infos.py --data_path data/v2x_real --save_path data/v2x_real --splits train val')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
