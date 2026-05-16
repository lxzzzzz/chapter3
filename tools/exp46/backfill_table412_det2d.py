import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

from common import EXP_DIR, ROOT_DIR, ROW_VARIANTS, dataset_specs, metrics_path, write_metrics

sys.path.insert(0, str(ROOT_DIR / 'pcdet' / 'datasets' / 'kitti'))
sys.path.insert(0, str(ROOT_DIR / 'pcdet' / 'utils'))
from det2d_eval_utils import eval_det2d_map50  # noqa: E402
import calibration_kitti  # noqa: E402


TABLE = '4-12'


def valid_bbox_mask(bboxes):
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0,), dtype=bool)
    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    return (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])


def make_calib(info):
    return calibration_kitti.Calibration(info['calib'])


def boxes3d_lidar_to_kitti_camera(boxes_lidar, calib):
    xyz_lidar = boxes_lidar[:, 0:3].copy()
    l, w, h = boxes_lidar[:, 3:4], boxes_lidar[:, 4:5], boxes_lidar[:, 5:6]
    heading = boxes_lidar[:, 6:7]
    xyz_lidar[:, 2] -= h[:, 0] / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    return np.concatenate([xyz_cam, l, h, w, -heading - np.pi / 2], axis=1)


def boxes3d_to_corners3d_kitti_camera(boxes_camera):
    boxes_num = boxes_camera.shape[0]
    l, h, w = boxes_camera[:, 3], boxes_camera[:, 4], boxes_camera[:, 5]

    x_corners = np.array([0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], dtype=np.float32)
    y_corners = np.array([0, 0, 0, 0, -1, -1, -1, -1], dtype=np.float32)
    z_corners = np.array([0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5], dtype=np.float32)
    x_corners = x_corners.reshape(1, 8).repeat(boxes_num, axis=0) * l.reshape(-1, 1)
    y_corners = y_corners.reshape(1, 8).repeat(boxes_num, axis=0) * h.reshape(-1, 1)
    z_corners = z_corners.reshape(1, 8).repeat(boxes_num, axis=0) * w.reshape(-1, 1)

    ry = boxes_camera[:, 6]
    zeros = np.zeros(ry.size, dtype=np.float32)
    ones = np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([
        [np.cos(ry), zeros, np.sin(ry)],
        [zeros, ones, zeros],
        [-np.sin(ry), zeros, np.cos(ry)],
    ])
    rot_mat = np.transpose(rot_list, (2, 0, 1))
    corners = np.concatenate(
        [x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1), z_corners.reshape(-1, 8, 1)],
        axis=2,
    )
    corners = np.matmul(corners, rot_mat)
    corners += boxes_camera[:, 0:3].reshape(-1, 1, 3)
    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes_camera, calib, image_shape):
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes_camera)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)
    min_uv = np.min(corners_in_image, axis=1)
    max_uv = np.max(corners_in_image, axis=1)
    boxes2d = np.concatenate([min_uv, max_uv], axis=1)
    boxes2d[:, 0] = np.clip(boxes2d[:, 0], a_min=0, a_max=image_shape[1] - 1)
    boxes2d[:, 1] = np.clip(boxes2d[:, 1], a_min=0, a_max=image_shape[0] - 1)
    boxes2d[:, 2] = np.clip(boxes2d[:, 2], a_min=0, a_max=image_shape[1] - 1)
    boxes2d[:, 3] = np.clip(boxes2d[:, 3], a_min=0, a_max=image_shape[0] - 1)
    return boxes2d


def project_lidar_boxes(boxes_lidar, calib, image_shape):
    boxes_lidar = np.asarray(boxes_lidar, dtype=np.float32).reshape(-1, 7)
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes_camera = boxes3d_lidar_to_kitti_camera(boxes_lidar, calib)
    return boxes3d_kitti_camera_to_imageboxes(
        boxes_camera, calib, image_shape=np.asarray(image_shape, dtype=np.int32)
    ).astype(np.float32)


def remap_to_classes(names, class_names, default_class='Car'):
    names = np.asarray(names)
    if len(names) == 0:
        return names
    class_set = set(class_names)
    return np.asarray([name if name in class_set else default_class for name in names])


def build_gt_annos(infos, class_names):
    gt_annos = []
    for info in infos:
        anno = copy.deepcopy(info.get('annos', {}))
        names = remap_to_classes(anno.get('name', []), class_names)
        boxes_lidar = np.asarray(anno.get('gt_boxes_lidar', np.zeros((0, 7))), dtype=np.float32).reshape(-1, 7)
        calib = make_calib(info)
        image_shape = info['image']['image_shape']
        anno['name'] = names
        anno['bbox'] = project_lidar_boxes(boxes_lidar, calib, image_shape)
        gt_annos.append(anno)
    return gt_annos


def build_det_annos(det_annos, infos):
    projected = []
    for det_anno, info in zip(det_annos, infos):
        anno = copy.deepcopy(det_anno)
        boxes_lidar = anno.get('boxes_lidar', np.zeros((0, 7), dtype=np.float32))
        calib = make_calib(info)
        image_shape = info['image']['image_shape']
        anno['bbox'] = project_lidar_boxes(boxes_lidar, calib, image_shape)
        projected.append(anno)
    return projected


def latest_result_pkl(row):
    candidates = sorted(
        ROOT_DIR.glob(f'output/**/exp46/{TABLE}/{row}/**/result.pkl'),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_json(path):
    import json

    with open(path, 'r') as f:
        return json.load(f)


def rows_from_args(rows):
    if rows:
        return rows
    return [row for row in ROW_VARIANTS if row.startswith('C')]


def main():
    parser = argparse.ArgumentParser(description='Backfill Table 4-12 2D AP@0.50 from saved result.pkl files.')
    parser.add_argument('--dataset', default='mths', choices=['mths', 'dair', 'v2x_real'])
    parser.add_argument('--rows', nargs='*', default=None, help='default: all C rows')
    parser.add_argument('--info-path', default=None, help='override eval info pickle path')
    args = parser.parse_args()

    spec = dataset_specs()[args.dataset]
    info_path = Path(args.info_path) if args.info_path else ROOT_DIR / spec.data_path / spec.info_path['test'][0]
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    class_names = spec.class_names
    gt_annos = build_gt_annos(infos, class_names)
    gt_valid = sum(int(valid_bbox_mask(anno.get('bbox')).sum()) for anno in gt_annos)
    if gt_valid == 0:
        raise RuntimeError(f'No valid projected GT 2D boxes from {info_path}')

    for row in rows_from_args(args.rows):
        result_path = latest_result_pkl(row)
        if result_path is None:
            print(f'{row}: skip, no result.pkl found')
            continue

        with open(result_path, 'rb') as f:
            det_annos = pickle.load(f)
        if len(det_annos) != len(infos):
            print(f'{row}: skip, result/info length mismatch {len(det_annos)} != {len(infos)}')
            continue

        projected_det_annos = build_det_annos(det_annos, infos)
        result_str, result_dict = eval_det2d_map50(gt_annos, projected_det_annos, class_names)

        metric_file = metrics_path(TABLE, row)
        if metric_file.exists():
            payload = load_json(metric_file)
        else:
            payload = {'table': TABLE, 'row': row, 'dataset': args.dataset}

        payload.update(result_dict)
        payload.setdefault('table', TABLE)
        payload.setdefault('row', row)
        payload.setdefault('dataset', args.dataset)
        payload['det2d_backfill_result_pkl'] = str(result_path)
        payload['det2d_backfill_info_path'] = str(info_path)

        summary_path = metric_file.parent / 'eval_summary.txt'
        summary_text = summary_path.read_text(errors='ignore') if summary_path.exists() else ''
        if '2D Detection AP@0.50:' in summary_text:
            summary_text = summary_text.split('2D Detection AP@0.50:')[0].rstrip() + '\n'
        payload['eval_summary'] = summary_text.rstrip() + '\n\n' + result_str

        out_path = write_metrics(TABLE, row, payload)
        det2d = result_dict.get('det2d_map_0.5')
        det2d_text = f'{det2d:.4f}' if det2d is not None else 'N/A'
        print(f'{row}: det2d_map_0.5={det2d_text} wrote {out_path}')

    table_path = EXP_DIR / TABLE / 'table.md'
    print(f'table: {table_path}')


if __name__ == '__main__':
    main()
