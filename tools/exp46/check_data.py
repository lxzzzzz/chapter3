import argparse
import os
import pickle
from pathlib import Path

from common import ROOT_DIR, dataset_specs


def ensure_link(link_path, target_path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return
    os.symlink(target_path, link_path)


def prepare_overlay(spec):
    local = ROOT_DIR / spec.data_path
    if not spec.external_path:
        return
    external = Path(spec.external_path)
    if not external.exists():
        return
    if spec.key == 'dair':
        if not local.exists() and not local.is_symlink():
            ensure_link(local, external)
        elif local.is_dir() and not local.is_symlink():
            for name in ('ImageSets', 'training'):
                ensure_link(local / name, external / name)
        return
    local.mkdir(parents=True, exist_ok=True)
    for name in ('ImageSets', 'training'):
        ensure_link(local / name, external / name)


def required_local_paths(spec):
    local = ROOT_DIR / spec.data_path
    if spec.key == 'dair':
        return [
            local / 'ImageSets',
            local / 'training' / 'velodyne',
            local / 'training' / 'image_2',
            local / 'training' / 'calib',
        ]
    if spec.key == 'mths':
        return [
            local / 'ImageSets',
            local / 'training' / 'velodyne',
            local / 'training' / 'image_02',
            local / 'training' / 'calib',
        ]
    if spec.key == 'v2x_real':
        return [
            local / 'ImageSets',
            local / 'training' / 'velodyne',
            local / 'training' / 'image_02',
            local / 'training' / 'calib',
        ]
    return [local]


def main():
    parser = argparse.ArgumentParser(description='Check Chapter 4.6 dataset roots and info files.')
    parser.add_argument('--no-link', action='store_true', help='only check paths, do not create local symlink overlays')
    parser.add_argument('--strict-external', action='store_true', help='fail when ROAD_ROOT external dataset roots are missing')
    args = parser.parse_args()

    ok = True
    for spec in dataset_specs().values():
        if not args.no_link:
            prepare_overlay(spec)
        local = ROOT_DIR / spec.data_path
        external = Path(spec.external_path) if spec.external_path else None
        external_exists = external.exists() if external is not None else False
        external_msg = external if external is not None else 'not configured'
        print(f'[{spec.key}] external={external_msg}')
        print(f'[{spec.key}] local={local}')
        if external is None:
            print('  external root not configured; using local project data only')
            ok = ok and not args.strict_external
        elif not external_exists:
            print('  MISSING external root (allowed if local data dirs below are OK)')
            ok = ok and not args.strict_external
        else:
            print('  external root OK')
        for path in required_local_paths(spec):
            exists = path.exists()
            broken = path.is_symlink() and not exists
            suffix = 'BROKEN_SYMLINK' if broken else ('OK' if exists else 'MISSING')
            print(f'  data: {path} {suffix}')
            ok = ok and exists
        if spec.key == 'v2x_real':
            image_dirs = [
                local / 'training' / name
                for name in ('image_02', 'image_03', 'image_04', 'image_05')
            ]
            available = [path.name for path in image_dirs if path.exists()]
            if available:
                print(f'  optional images: {", ".join(available)} OK')
            else:
                print('  optional images: none found; V2X-Real will use LiDAR-only configs')
        for split, names in spec.info_path.items():
            for name in names:
                path = local / name
                exists = path.exists()
                print(f'  {split}: {path} {"OK" if exists else "MISSING"}')
                ok = ok and exists
                if exists and spec.key == 'v2x_real':
                    try:
                        with open(path, 'rb') as f:
                            infos = pickle.load(f)
                        image_path = infos[0].get('image', {}).get('image_path', '') if infos else ''
                        image_ok = bool(image_path)
                        print(f'  {split} image_path in info: {image_path if image_ok else "MISSING"}')
                        ok = ok and image_ok
                    except Exception as exc:
                        print(f'  {split} image_path check failed: {exc!r}')
                        ok = False
        if external is not None and not external_exists:
            print(f'  optional external fix: export ROAD_ROOT=/path/to/Roadside')
        print(f'  local dataset root: {local}')
        if spec.key == 'mths':
            print('  build if missing: python tools/create_v2x_xian_infos.py --data_path data/v2x_xian --save_path data/v2x_xian --splits train val')
        elif spec.key == 'v2x_real':
            print('  rebuild for multimodal: python tools/create_v2x_real_infos.py --data_path data/v2x_real --save_path data/v2x_real --splits train val --reference_camera 02')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
