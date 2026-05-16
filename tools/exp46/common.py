import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    'F1': 'full', 'F2': 'full', 'F3': 'full', 'F4': 'full', 'F5': 'full',
    'G1': 'full', 'G2': 'full', 'G3': 'full', 'G4': 'full',
}

ROW_CFG_OVERRIDES = {
    'F1': [('MODEL.ENABLE_AUX_LOSS', False), ('MODEL.AUX_LOSS_WEIGHT', 0.0)],
    'F2': [('MODEL.ENABLE_AUX_LOSS', True), ('MODEL.AUX_LOSS_WEIGHT', 0.05)],
    'F3': [('MODEL.ENABLE_AUX_LOSS', True), ('MODEL.AUX_LOSS_WEIGHT', 0.10)],
    'F4': [('MODEL.ENABLE_AUX_LOSS', True), ('MODEL.AUX_LOSS_WEIGHT', 0.20)],
    'F5': [('MODEL.ENABLE_AUX_LOSS', True), ('MODEL.AUX_LOSS_WEIGHT', 0.30)],
    'G1': [('MODEL.BACKBONE_3D.DEFORMABLE_NUM_POINTS', 1)],
    'G2': [('MODEL.BACKBONE_3D.DEFORMABLE_NUM_POINTS', 4)],
    'G3': [('MODEL.BACKBONE_3D.DEFORMABLE_NUM_POINTS', 8)],
    'G4': [('MODEL.BACKBONE_3D.DEFORMABLE_NUM_POINTS', 16)],
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
    'TransFusion', 'BEVFusion', 'SparseFusion_DeepInteraction', 'Ours',
]

METHOD_VARIANTS = {
    'PointPillars': 'pointpillar',
    'SECOND': 'second',
    'CenterPoint': 'centerpoint',
    'VoxelNeXt': 'voxelnext',
    'TransFusion': 'transfusion',
    'BEVFusion': 'bevfusion',
    'SparseFusion_DeepInteraction': 'all_sample',
    'Ours': 'full',
}

METHOD_LABELS = {
    'SparseFusion_DeepInteraction': 'SparseFusion/DeepInteraction',
}

METHOD_MODALITY = {
    'PointPillars': 'LiDAR',
    'SECOND': 'LiDAR',
    'CenterPoint': 'LiDAR',
    'VoxelNeXt': 'LiDAR',
    'TransFusion': 'LiDAR+Camera',
    'BEVFusion': 'LiDAR+Camera',
    'SparseFusion_DeepInteraction': 'LiDAR+Camera',
    'Ours': 'LiDAR+Camera+2D backend',
}

METHOD_FUSION_TYPE = {
    'PointPillars': 'Pillar',
    'SECOND': 'Voxel',
    'CenterPoint': 'Voxel/BEV',
    'VoxelNeXt': 'Sparse voxel',
    'TransFusion': 'Query fusion',
    'BEVFusion': 'BEV fusion',
    'SparseFusion_DeepInteraction': 'Sparse interaction',
    'Ours': 'VoxelNeXt',
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


def align_range_to_voxel_size(point_cloud_range, voxel_size):
    aligned = [float(x) for x in point_cloud_range]
    mins = aligned[:3]
    maxs = aligned[3:6]
    for axis, voxel in enumerate(voxel_size):
        size = (maxs[axis] - mins[axis]) / float(voxel)
        aligned_cells = int(np.ceil(size - 1e-6))
        maxs[axis] = mins[axis] + aligned_cells * float(voxel)
    return mins + maxs


def align_range_to_grid_multiple(point_cloud_range, voxel_size, grid_multiple):
    aligned = align_range_to_voxel_size(point_cloud_range, voxel_size)
    mins = aligned[:3]
    maxs = aligned[3:6]
    for axis in (0, 1):
        voxel = float(voxel_size[axis])
        cells = int(round((maxs[axis] - mins[axis]) / voxel))
        aligned_cells = int(np.ceil(cells / float(grid_multiple)) * grid_multiple)
        maxs[axis] = mins[axis] + aligned_cells * voxel
    return mins + maxs


def set_point_cloud_range(cfg, point_cloud_range):
    data_cfg = cfg.setdefault('DATA_CONFIG', {})
    data_cfg['POINT_CLOUD_RANGE'] = point_cloud_range

    model_cfg = cfg.setdefault('MODEL', {})
    model_cfg['POINT_CLOUD_RANGE'] = point_cloud_range
    if model_cfg.get('BACKBONE_3D'):
        model_cfg['BACKBONE_3D']['POINT_CLOUD_RANGE'] = point_cloud_range
    dense = model_cfg.get('DENSE_HEAD', {})
    if dense.get('POST_PROCESSING'):
        dense['POST_PROCESSING']['POST_CENTER_LIMIT_RANGE'] = point_cloud_range
        dense['POST_PROCESSING']['POST_CENTER_RANGE'] = point_cloud_range


def ensure_data_processors(cfg):
    data_cfg = cfg.setdefault('DATA_CONFIG', {})
    processors = data_cfg.get('DATA_PROCESSOR')
    if not processors and data_cfg.get('_BASE_CONFIG_'):
        base_cfg = load_yaml(data_cfg['_BASE_CONFIG_'])
        processors = deepcopy(base_cfg.get('DATA_PROCESSOR', []))
        data_cfg['DATA_PROCESSOR'] = processors
    return processors or []


def set_voxel_size(cfg, voxel_size):
    data_cfg = cfg.setdefault('DATA_CONFIG', {})
    for processor in ensure_data_processors(cfg):
        if processor.get('NAME') == 'transform_points_to_voxels':
            processor['VOXEL_SIZE'] = [float(x) for x in voxel_size]

    model_cfg = cfg.setdefault('MODEL', {})
    model_cfg['VOXEL_SIZE'] = [float(x) for x in voxel_size]
    if model_cfg.get('BACKBONE_3D'):
        model_cfg['BACKBONE_3D']['VOXEL_SIZE'] = [float(x) for x in voxel_size]


def make_compressed_3d_voxel_size(spec, target_z_cells=40):
    z_extent = float(spec.point_cloud_range[5] - spec.point_cloud_range[2])
    return [float(spec.voxel_size[0]), float(spec.voxel_size[1]), z_extent / float(target_z_cells)]


def adapt_compressed_3d_grid(cfg, spec, feature_map_stride=8):
    voxel_size = make_compressed_3d_voxel_size(spec)
    point_cloud_range = align_range_to_grid_multiple(
        spec.point_cloud_range,
        voxel_size,
        grid_multiple=feature_map_stride,
    )
    set_voxel_size(cfg, voxel_size)
    set_point_cloud_range(cfg, point_cloud_range)
    if cfg.get('MODEL', {}).get('MAP_TO_BEV'):
        cfg['MODEL']['MAP_TO_BEV']['NUM_BEV_FEATURES'] = 256


def limit_max_voxels(cfg, max_voxels=None):
    if max_voxels is None:
        return
    max_voxels = int(max_voxels)
    data_cfg = cfg.get('DATA_CONFIG', {})
    processors = ensure_data_processors(cfg)
    for processor in processors or []:
        if processor.get('NAME') == 'transform_points_to_voxels':
            processor['MAX_NUMBER_OF_VOXELS'] = {
                'train': max_voxels,
                'test': max_voxels,
            }


def set_nested(mapping, keys, value):
    cur = mapping
    for key in keys[:-1]:
        if cur.get(key) is None:
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def set_dotted(mapping, dotted_key, value):
    set_nested(mapping, dotted_key.split('.'), value)


def apply_row_overrides(cfg, row=None):
    for dotted_key, value in ROW_CFG_OVERRIDES.get(row, []):
        set_dotted(cfg, dotted_key, value)
    if row in ROW_CFG_OVERRIDES:
        cfg.setdefault('EXP46', {})['row_overrides'] = {
            key: value for key, value in ROW_CFG_OVERRIDES[row]
        }
    return cfg


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
    point_cloud_range = align_range_to_voxel_size(spec.point_cloud_range, spec.voxel_size)
    data_cfg = cfg.setdefault('DATA_CONFIG', {})
    data_cfg['_BASE_CONFIG_'] = spec.fusion_base if use_images else spec.lidar_base
    data_cfg['DATA_PATH'] = spec.data_path
    set_point_cloud_range(cfg, point_cloud_range)
    data_cfg['INFO_PATH'] = deepcopy(spec.info_path)
    data_cfg['GET_ITEM_LIST'] = ['points', 'images', 'calib_matrices'] if use_images else ['points']
    for processor in data_cfg.get('DATA_PROCESSOR', []):
        if processor.get('NAME') == 'transform_points_to_voxels':
            processor['VOXEL_SIZE'] = spec.voxel_size
    if spec.key in ('mths', 'v2x_real'):
        data_cfg['MAP_CLASS_TO_KITTI'] = {name: name for name in spec.class_names}
        data_cfg['DEFAULT_CLASS_NAME'] = 'Car'

    model_cfg = cfg.setdefault('MODEL', {})
    model_cfg['VOXEL_SIZE'] = spec.voxel_size
    if model_cfg.get('BACKBONE_3D'):
        model_cfg['BACKBONE_3D']['VOXEL_SIZE'] = spec.voxel_size
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


def make_cfg(variant, dataset, epochs, row=None, max_voxels=None):
    spec = dataset_specs()[dataset]
    if variant == 'pointpillar':
        cfg = load_yaml('tools/cfgs/kitti_models/pointpillar.yaml')
        adapt_dataset(cfg, spec, use_images=False)
        for processor in cfg.get('DATA_CONFIG', {}).get('DATA_PROCESSOR', []):
            if processor.get('NAME') == 'transform_points_to_voxels':
                processor['VOXEL_SIZE'] = [spec.voxel_size[0], spec.voxel_size[1], max(spec.point_cloud_range[5] - spec.point_cloud_range[2], 1.0)]
                point_cloud_range = align_range_to_grid_multiple(
                    spec.point_cloud_range,
                    processor['VOXEL_SIZE'],
                    grid_multiple=2,
                )
                set_point_cloud_range(cfg, point_cloud_range)
    elif variant == 'second':
        cfg = load_yaml('tools/cfgs/kitti_models/second.yaml')
        adapt_dataset(cfg, spec, use_images=False)
        adapt_compressed_3d_grid(cfg, spec, feature_map_stride=8)
    elif variant == 'centerpoint':
        cfg = load_yaml('tools/cfgs/once_models/centerpoint.yaml')
        adapt_dataset(cfg, spec, use_images=False)
        adapt_compressed_3d_grid(cfg, spec, feature_map_stride=8)
    elif variant == 'transfusion':
        cfg = make_transfusion_cfg(spec)
        adapt_compressed_3d_grid(cfg, spec, feature_map_stride=8)
    else:
        cfg = make_voxelnext_cfg(spec, variant)

    adapt_classes(cfg, spec.class_names)
    remove_gt_sampling(cfg)
    limit_max_voxels(cfg, max_voxels=max_voxels)
    cfg.setdefault('OPTIMIZATION', {})['NUM_EPOCHS'] = int(epochs)
    cfg['OPTIMIZATION'].setdefault('BATCH_SIZE_PER_GPU', 4)
    apply_row_overrides(cfg, row=row)
    return cfg


def write_cfg(variant, dataset, epochs, name, row=None, max_voxels=None):
    cfg = make_cfg(variant, dataset, epochs, row=row, max_voxels=max_voxels)
    cfg_path = CFG_DIR / f'{name}.yaml'
    dump_yaml(cfg, cfg_path)
    return cfg_path


def run_train(cfg_path, table, row, epochs, workers=4, batch_size=None, use_amp=False, extra_args=None):
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
    if batch_size is not None:
        cmd.extend(['--batch_size', str(batch_size)])
    if use_amp:
        cmd.append('--use_amp')
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=ROOT_DIR)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return cmd


def metrics_path(table, row):
    return EXP_DIR / table / row / 'metrics.json'


def latest_eval_log(table=None, row=None):
    patterns = ['log_eval_*.txt', 'train_*.log']
    if table is not None and row is not None:
        logs = []
        for pattern in patterns:
            logs.extend((ROOT_DIR / 'output').glob(f'**/exp46/{table}/{row}/**/{pattern}'))
        logs = sorted(logs, key=lambda p: p.stat().st_mtime)
        if logs:
            return logs[-1]
    logs = []
    for pattern in patterns:
        logs.extend((ROOT_DIR / 'output').glob(f'**/{pattern}'))
    logs = sorted(logs, key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


PRIMARY_THRESHOLDS = {
    'Car': '0.70,0.50,0.50',
    'Pedestrian': '0.50,0.25,0.25',
    'Cyclist': '0.50,0.25,0.25',
}

DISTANCE_KEYS = ['0-40', '40-80', '80-120']


def _norm_threshold(text):
    return ','.join(part.strip() for part in text.split(','))


def _parse_ap_blocks(text):
    blocks = {'overall': {}}
    summary_lines = []
    context = 'overall'
    current = None
    header_re = re.compile(r'^(Car|Pedestrian|Cyclist|Van|Truck)\s+(AP(?:_R40)?)@([0-9., ]+):')
    metric_re = re.compile(r'^(bbox|bev|3d|aos)\s+AP:([0-9., -]+)')
    range_re = re.compile(r'^--- Distance Range:\s*(\d+)m to (\d+)m ---')

    for raw_line in text.splitlines():
        line = raw_line.strip()
        range_match = range_re.match(line)
        if range_match:
            context = f'{range_match.group(1)}-{range_match.group(2)}'
            blocks.setdefault(context, {})
            summary_lines.append(line)
            current = None
            continue

        header_match = header_re.match(line)
        if header_match:
            cls_name, family, threshold = header_match.groups()
            threshold = _norm_threshold(threshold)
            current = (context, cls_name, family, threshold)
            blocks.setdefault(context, {}).setdefault(cls_name, {}).setdefault(family, {}).setdefault(threshold, {})
            summary_lines.append(line)
            continue

        metric_match = metric_re.match(line)
        if metric_match and current is not None:
            metric_name, values = metric_match.groups()
            nums = [float(item.strip()) for item in values.split(',') if item.strip()]
            blocks[current[0]][current[1]][current[2]][current[3]][metric_name] = nums
            summary_lines.append(line)
            continue

        if line.startswith('=') or 'Distance-based Evaluation' in line or '2D AP@0.50' in line or '2D mAP@0.50' in line:
            summary_lines.append(line)

    return blocks, '\n'.join(summary_lines).strip() + '\n'


def _select_class_value(blocks, context, cls_name):
    cls_blocks = blocks.get(context, {}).get(cls_name, {})
    r40_blocks = cls_blocks.get('AP_R40', {})
    threshold = PRIMARY_THRESHOLDS.get(cls_name)
    metric_block = r40_blocks.get(threshold, {}) if threshold else {}
    if '3d' not in metric_block:
        for candidate in r40_blocks.values():
            if '3d' in candidate:
                metric_block = candidate
                break
    values = metric_block.get('3d')
    if not values or len(values) < 2:
        return None
    return float(values[1])


def _mean_available(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _build_table_values(blocks, class_names):
    table_values = {
        'metric_definition': '3D AP_R40 moderate; Car uses AP_R40@0.70,0.50,0.50, Pedestrian/Cyclist use AP_R40@0.50,0.25,0.25',
        'per_class': {},
    }

    for cls_name in class_names:
        cls_values = {'overall': _select_class_value(blocks, 'overall', cls_name)}
        for dist_key in DISTANCE_KEYS:
            cls_values[dist_key] = _select_class_value(blocks, dist_key, cls_name)
        table_values['per_class'][cls_name] = cls_values

    table_values['overall'] = _mean_available([
        table_values['per_class'][cls_name]['overall'] for cls_name in class_names
    ])
    for dist_key in DISTANCE_KEYS:
        table_values[dist_key] = _mean_available([
            table_values['per_class'][cls_name][dist_key] for cls_name in class_names
        ])
    return table_values


def parse_eval_log(log_path, dataset=None):
    metrics = {}
    if log_path is None or not log_path.exists():
        return metrics
    text = log_path.read_text(errors='ignore')
    for key, value in re.findall(r'(recall_(?:roi|rcnn)_[0-9.]+):\s+([0-9.]+)', text):
        metrics[key] = float(value)
    det2d_match = re.findall(r'2D mAP@0\.50:\s+([0-9.]+)', text)
    if det2d_match:
        metrics['det2d_map_0.5'] = float(det2d_match[-1])
    for cls_name, value in re.findall(r'(Car|Pedestrian|Cyclist)\s+2D AP@0\.50:\s+([0-9.]+)', text):
        metrics[f'det2d_{cls_name.lower()}_ap_0.5'] = float(value)
    blocks, summary_text = _parse_ap_blocks(text)
    class_names = dataset_specs()[dataset].class_names if dataset else sorted(blocks.get('overall', {}).keys())
    table_values = _build_table_values(blocks, class_names)
    metrics['ap_blocks'] = blocks
    metrics['eval_summary'] = summary_text
    metrics['table_values'] = table_values
    metrics['ap_3d'] = table_values.get('overall')
    metrics['ap_0_40'] = table_values.get('0-40')
    metrics['ap_40_80'] = table_values.get('40-80')
    metrics['ap_80_120'] = table_values.get('80-120')
    for cls_name, cls_values in table_values.get('per_class', {}).items():
        prefix = cls_name.lower()
        metrics[f'{prefix}_ap_3d'] = cls_values.get('overall')
        metrics[f'{prefix}_ap_0_40'] = cls_values.get('0-40')
        metrics[f'{prefix}_ap_40_80'] = cls_values.get('40-80')
        metrics[f'{prefix}_ap_80_120'] = cls_values.get('80-120')
    return metrics


def write_metrics(table, row, payload):
    path = metrics_path(table, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = payload.pop('eval_summary', None)
    table_values = payload.get('table_values')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    if summary_text is not None:
        (path.parent / 'eval_summary.txt').write_text(summary_text)
    if table_values is not None:
        with open(path.parent / 'table_values.json', 'w') as f:
            json.dump(table_values, f, indent=2, sort_keys=True)
    write_table_markdown(table)
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

    if table in {'4-17', '4-18', '4-19', '4-20'}:
        order = {name: idx for idx, name in enumerate(COMPARE_METHODS)}
    elif table == '4-21':
        order = {name: idx for idx, name in enumerate(PROFILE_VARIANTS.keys())}
    else:
        order = {name: idx for idx, name in enumerate(ROW_VARIANTS.keys())}
    rows.sort(key=lambda item: (order.get(item.get('row'), 999), item.get('row', '')))
    return rows


def format_table_value(value):
    if isinstance(value, float):
        return f'{value:.4f}'
    if value is None:
        return ''
    return str(value)


def table_columns(table, rows):
    if table == '4-21':
        return ['row', 'variant', 'dataset', 'params', 'trainable_params', 'elapsed_sec', 'profile_error']

    columns = ['row']
    if table in {'4-17', '4-18', '4-19', '4-20'}:
        columns.extend(['method', 'modality', 'fusion_type'])
    else:
        columns.extend(['method', 'variant'])
    columns.extend([
        'dataset', 'epochs',
        'ap_3d', 'ap_0_40', 'ap_40_80', 'ap_80_120',
    ])
    if table == '4-12' and any(row.get('det2d_map_0.5') is not None for row in rows):
        columns.append('det2d_map_0.5')
    class_columns = [
        'car_ap_3d', 'pedestrian_ap_3d', 'cyclist_ap_3d',
        'car_ap_0_40', 'car_ap_40_80', 'car_ap_80_120',
        'pedestrian_ap_0_40', 'pedestrian_ap_40_80', 'pedestrian_ap_80_120',
        'cyclist_ap_0_40', 'cyclist_ap_40_80', 'cyclist_ap_80_120',
    ]
    for column in class_columns:
        if any(row.get(column) is not None for row in rows):
            columns.append(column)
    columns.extend(['recall_rcnn_0.3', 'recall_rcnn_0.5', 'recall_rcnn_0.7'])
    return columns


def build_table_markdown(table, rows):
    columns = table_columns(table, rows)
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(format_table_value(row.get(col)) for col in columns) + ' |')
    return '\n'.join(lines)


def write_table_markdown(table):
    rows = read_metrics(table)
    if not rows:
        return None
    table_path = EXP_DIR / table / 'table.md'
    table_path.write_text(build_table_markdown(table, rows) + '\n')
    return table_path
