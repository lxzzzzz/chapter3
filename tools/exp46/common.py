import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT_DIR / 'output' / 'exp46'
CFG_DIR = ROOT_DIR / 'tools' / 'cfgs' / 'exp46'


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    class_names: list
    data_path: str
    external_path: str
    lidar_base: str
    fusion_base: str
    point_cloud_range: list
    voxel_size: list
    info_path: dict


def road_root():
    value = os.environ.get('ROAD_ROOT', '').strip()
    return Path(value) if value else None


def external_dataset_path(roadside_root, *parts):
    if roadside_root is None:
        return ''
    return str(roadside_root.joinpath(*parts))


def dataset_specs():
    rr = road_root()
    return {
        'mths': DatasetSpec(
            key='mths',
            class_names=['Car'],
            data_path='data/v2x_xian',
            external_path=external_dataset_path(rr, 'V2X-xian-trainval-kitti'),
            lidar_base='tools/cfgs/dataset_configs/v2x_xian_lidar_det_dataset.yaml',
            fusion_base='tools/cfgs/dataset_configs/v2x_xian_fusion_det_dataset.yaml',
            point_cloud_range=[0, -20, -4.5, 120, 30, 10.5],
            voxel_size=[0.16, 0.16, 0.1],
            info_path={'train': ['tracking_infos_train.pkl'], 'test': ['tracking_infos_val.pkl']},
        ),
        'dair': DatasetSpec(
            key='dair',
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path='data/dair_v2x',
            external_path=external_dataset_path(rr, 'DAIR-V2X-I', 'DAIR-V2X-I-kitti'),
            lidar_base='tools/cfgs/dataset_configs/dair_v2x_lidar_dataset_100m.yaml',
            fusion_base='tools/cfgs/dataset_configs/dair_v2x_dataset_100m.yaml',
            point_cloud_range=[0, -51.2, -5, 102.4, 51.2, 3],
            voxel_size=[0.16, 0.16, 0.1],
            info_path={'train': ['dair_v2x_infos_train.pkl'], 'test': ['dair_v2x_infos_val.pkl']},
        ),
        'v2x_real': DatasetSpec(
            key='v2x_real',
            class_names=['Car'],
            data_path='data/v2x_real',
            external_path=external_dataset_path(rr, 'V2X-Real-trainval'),
            lidar_base='tools/cfgs/dataset_configs/v2x_real_lidar_det_dataset.yaml',
            fusion_base='tools/cfgs/dataset_configs/v2x_real_lidar_det_dataset.yaml',
            point_cloud_range=[-102.4, -40, -15, 102.4, 40, 15],
            voxel_size=[0.16, 0.16, 0.1],
            info_path={'train': ['detect_infos_train.pkl'], 'test': ['detect_infos_val.pkl']},
        ),
    }


ROW_VARIANTS = {
    'A1': 'voxelnext', 'A2': 'pvga', 'A3': 'direct', 'A4': 'deform',
    'A5': 'rule_route', 'A6': 'all_sample', 'A7': 'selective', 'A8': 'full',
    'B1': 'voxelnext', 'B2': 'pvga', 'B3': 'direct', 'B4': 'deform',
    'B5': 'rule_route', 'B6': 'all_sample', 'B7': 'selective', 'B8': 'full',
    'C1': 'voxelnext', 'C2': 'pvga', 'C3': 'direct', 'C4': 'deform',
    'C5': 'rule_route', 'C6': 'all_sample', 'C7': 'selective', 'C8': 'full',
    'D1': 'voxelnext', 'D2': 'pvga', 'D3': 'direct',
    'D4': 'rule_route', 'D5': 'selective', 'D6': 'full',
    'E1': 'voxelnext', 'E2': 'pvga', 'E3': 'deform',
    'E4': 'rule_route', 'E5': 'selective', 'E6': 'full',
    'F1': 'voxelnext', 'F2': 'direct', 'F3': 'deform', 'F4': 'selective', 'F5': 'full',
    'G1': 'voxelnext', 'G2': 'pvga', 'G3': 'selective', 'G4': 'full',
}

PROFILE_VARIANTS = {
    'voxelnext': 'voxelnext',
    'v43': 'pvga',
    'v43_v45': 'selective',
    'det2d': 'all_sample',
    'full': 'full',
}

COMPARE_METHODS = [
    'PointPillars', 'SECOND', 'CenterPoint', 'VoxelNeXt',
    'TransFusion', 'BEVFusion', 'SparseFusion', 'DeepInteraction', 'Ours',
]

METHOD_VARIANTS = {
    'PointPillars': 'pointpillar',
    'SECOND': 'second',
    'CenterPoint': 'centerpoint',
    'VoxelNeXt': 'voxelnext',
    'TransFusion': 'transfusion',
    'BEVFusion': 'bevfusion',
    'SparseFusion': 'direct',
    'DeepInteraction': 'all_sample',
    'Ours': 'full',
}


def load_yaml(path):
    with open(ROOT_DIR / path, 'r') as f:
        return yaml.safe_load(f) or {}


def dump_yaml(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def remove_gt_sampling(cfg):
    aug = cfg.get('DATA_CONFIG', {}).get('DATA_AUGMENTOR')
    if not aug:
        return
    aug['AUG_CONFIG_LIST'] = [
        item for item in aug.get('AUG_CONFIG_LIST', [])
        if item.get('NAME') != 'gt_sampling'
    ]


def set_nested(mapping, keys, value):
    cur = mapping
    for key in keys[:-1]:
        if cur.get(key) is None:
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def adapt_classes(cfg, classes):
    cfg['CLASS_NAMES'] = classes
    dense = cfg.get('MODEL', {}).get('DENSE_HEAD', {})
    if 'CLASS_NAMES_EACH_HEAD' in dense:
        dense['CLASS_NAMES_EACH_HEAD'] = [classes]
    if 'NUM_CLASSES' in dense:
        dense['NUM_CLASSES'] = len(classes)
    anchors = dense.get('ANCHOR_GENERATOR_CONFIG')
    if anchors:
        dense['ANCHOR_GENERATOR_CONFIG'] = [item for item in anchors if item.get('class_name') in classes]


def adapt_dataset(cfg, spec, use_images):
    data_cfg = cfg.setdefault('DATA_CONFIG', {})
    data_cfg['_BASE_CONFIG_'] = spec.fusion_base if use_images else spec.lidar_base
    data_cfg['DATA_PATH'] = spec.data_path
    data_cfg['POINT_CLOUD_RANGE'] = spec.point_cloud_range
    data_cfg['INFO_PATH'] = deepcopy(spec.info_path)
    data_cfg['GET_ITEM_LIST'] = ['points', 'images', 'calib_matrices'] if use_images else ['points']
    for processor in data_cfg.get('DATA_PROCESSOR', []):
        if processor.get('NAME') == 'transform_points_to_voxels':
            processor['VOXEL_SIZE'] = spec.voxel_size
    if spec.key in ('mths', 'v2x_real'):
        data_cfg['MAP_CLASS_TO_KITTI'] = {name: name for name in spec.class_names}
        data_cfg['DEFAULT_CLASS_NAME'] = 'Car'

    model_cfg = cfg.setdefault('MODEL', {})
    model_cfg['POINT_CLOUD_RANGE'] = spec.point_cloud_range
    model_cfg['VOXEL_SIZE'] = spec.voxel_size
    if model_cfg.get('BACKBONE_3D'):
        model_cfg['BACKBONE_3D']['POINT_CLOUD_RANGE'] = spec.point_cloud_range
        model_cfg['BACKBONE_3D']['VOXEL_SIZE'] = spec.voxel_size
    dense = model_cfg.get('DENSE_HEAD', {})
    if dense.get('POST_PROCESSING'):
        dense['POST_PROCESSING']['POST_CENTER_LIMIT_RANGE'] = spec.point_cloud_range
        dense['POST_PROCESSING']['POST_CENTER_RANGE'] = spec.point_cloud_range
    model_cfg.setdefault('POST_PROCESSING', {})['EVAL_METRIC'] = 'kitti'


def make_voxelnext_cfg(spec, variant):
    if variant == 'voxelnext':
        cfg = load_yaml({
            'mths': 'tools/cfgs/dair_v2x_models/voxelnext_v2x_xian_lidar.yaml',
            'dair': 'tools/cfgs/dair_v2x_models/voxelnext_dair_lidar_100m.yaml',
            'v2x_real': 'tools/cfgs/custom_models/voxelnext_v2x_real_lidar.yaml',
        }[spec.key])
        adapt_dataset(cfg, spec, use_images=False)
        return cfg

    cfg = load_yaml({
        'mths': 'tools/cfgs/dair_v2x_models/fusion_voxelnext_v2x_xian.yaml',
        'dair': 'tools/cfgs/dair_v2x_models/selective_moe_voxelnext_100m.yaml',
        'v2x_real': 'tools/cfgs/dair_v2x_models/fusion_voxelnext_v2x_xian.yaml',
    }[spec.key])
    adapt_dataset(cfg, spec, use_images=True)
    model = cfg['MODEL']

    if variant in ('pvga', 'bevfusion'):
        model['NAME'] = 'FusionVoxelNeXt'
        model['BACKBONE_3D']['NAME'] = 'VoxelNeXtPVGA'
    else:
        model['NAME'] = 'RoadsideMultimodalVoxelNeXt'
        model['BACKBONE_3D']['NAME'] = {
            'direct': 'VoxelNeXtDSFADirect',
            'deform': 'VoxelNeXtDSFADeform',
            'rule_route': 'VoxelNeXtDSFARuleRoute',
            'all_sample': 'VoxelNeXtDSFAAllSample',
            'selective': 'VoxelNeXtDSFA',
            'full': 'VoxelNeXtDSFA',
        }[variant]
        model['ENABLE_AUX_LOSS'] = True
        model['AUX_LOSS_WEIGHT'] = 0.1
        if variant == 'full':
            model['BACKBONE_3D']['RECTIFY_RATIO'] = 0.35
            model['BACKBONE_3D']['DEFORMABLE_NUM_POINTS'] = 21
    return cfg


def make_transfusion_cfg(spec):
    cfg = load_yaml('tools/cfgs/nuscenes_models/transfusion_lidar.yaml')
    adapt_dataset(cfg, spec, use_images=False)
    dense = cfg['MODEL']['DENSE_HEAD']
    dense['NAME'] = 'TransFusionHeadKITTI'
    dense['NUM_CLASSES'] = len(spec.class_names)
    dense['SEPARATE_HEAD_CFG']['HEAD_ORDER'] = ['center', 'height', 'dim', 'rot']
    dense['SEPARATE_HEAD_CFG']['HEAD_DICT'] = {
        'center': {'out_channels': 2, 'num_conv': 2},
        'height': {'out_channels': 1, 'num_conv': 2},
        'dim': {'out_channels': 3, 'num_conv': 2},
        'rot': {'out_channels': 2, 'num_conv': 2},
    }
    dense['LOSS_CONFIG']['LOSS_WEIGHTS']['code_weights'] = [1.0] * 8
    dense['TARGET_ASSIGNER_CONFIG']['DATASET'] = 'kitti'
    cfg['MODEL']['POST_PROCESSING']['EVAL_METRIC'] = 'kitti'
    return cfg


def make_cfg(variant, dataset, epochs):
    spec = dataset_specs()[dataset]
    if variant == 'pointpillar':
        cfg = load_yaml('tools/cfgs/kitti_models/pointpillar.yaml')
        adapt_dataset(cfg, spec, use_images=False)
        for processor in cfg.get('DATA_CONFIG', {}).get('DATA_PROCESSOR', []):
            if processor.get('NAME') == 'transform_points_to_voxels':
                processor['VOXEL_SIZE'] = [spec.voxel_size[0], spec.voxel_size[1], max(spec.point_cloud_range[5] - spec.point_cloud_range[2], 1.0)]
    elif variant == 'second':
        cfg = load_yaml('tools/cfgs/kitti_models/second.yaml')
        adapt_dataset(cfg, spec, use_images=False)
    elif variant == 'centerpoint':
        cfg = load_yaml('tools/cfgs/once_models/centerpoint.yaml')
        adapt_dataset(cfg, spec, use_images=False)
    elif variant == 'transfusion':
        cfg = make_transfusion_cfg(spec)
    else:
        cfg = make_voxelnext_cfg(spec, variant)

    adapt_classes(cfg, spec.class_names)
    remove_gt_sampling(cfg)
    cfg.setdefault('OPTIMIZATION', {})['NUM_EPOCHS'] = int(epochs)
    cfg['OPTIMIZATION'].setdefault('BATCH_SIZE_PER_GPU', 4)
    return cfg


def write_cfg(variant, dataset, epochs, name):
    cfg = make_cfg(variant, dataset, epochs)
    cfg_path = CFG_DIR / f'{name}.yaml'
    dump_yaml(cfg, cfg_path)
    return cfg_path


def run_train(cfg_path, table, row, epochs, workers=4, extra_args=None):
    cmd = [
        sys.executable, 'tools/train.py',
        '--cfg_file', str(cfg_path.relative_to(ROOT_DIR)),
        '--extra_tag', f'exp46/{table}/{row}',
        '--epochs', str(epochs),
        '--num_epochs_to_eval', '1',
        '--max_waiting_mins', '1',
        '--ckpt_save_interval', str(max(1, int(epochs))),
        '--workers', str(workers),
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=ROOT_DIR)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return cmd


def metrics_path(table, row):
    return EXP_DIR / table / row / 'metrics.json'


def latest_eval_log():
    logs = sorted((ROOT_DIR / 'output').glob('**/log_eval_*.txt'), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_eval_log(log_path):
    metrics = {}
    if log_path is None or not log_path.exists():
        return metrics
    text = log_path.read_text(errors='ignore')
    for key, value in re.findall(r'(recall_(?:roi|rcnn)_[0-9.]+):\s+([0-9.]+)', text):
        metrics[key] = float(value)
    for line in text.splitlines():
        if '3d AP' in line.lower() or 'bev AP' in line.lower() or 'bbox AP' in line.lower():
            nums = [float(x) for x in re.findall(r'(?<![A-Za-z])\d+\.\d+', line)]
            if nums:
                metrics.setdefault('ap_values', []).append(nums)
    if metrics.get('ap_values'):
        flat = [v for row in metrics['ap_values'] for v in row]
        metrics['ap_mean'] = sum(flat) / len(flat)
    return metrics


def write_metrics(table, row, payload):
    path = metrics_path(table, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def read_metrics(table):
    table_dir = EXP_DIR / table
    if not table_dir.exists():
        return []
    rows = []
    for path in sorted(table_dir.glob('*/metrics.json')):
        with open(path, 'r') as f:
            item = json.load(f)
        item.setdefault('row', path.parent.name)
        rows.append(item)
    return rows
