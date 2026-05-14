import copy

import numpy as np


DISTANCE_RANGES = [(0, 40), (40, 80), (80, 120)]


def _first_axis_filter(anno, mask):
    out = copy.deepcopy(anno)
    for key, value in list(out.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape:
            out[key] = value[mask]
    return out


def _bev_distance(anno):
    boxes = anno.get('boxes_lidar', None)
    if boxes is None:
        boxes = anno.get('gt_boxes_lidar', None)
    if boxes is not None and len(boxes) > 0:
        boxes = np.asarray(boxes)
        return np.linalg.norm(boxes[:, :2], axis=1)

    loc = anno.get('location', None)
    if loc is not None and len(loc) > 0:
        loc = np.asarray(loc)
        return np.sqrt(loc[:, 0] ** 2 + loc[:, 2] ** 2)

    return np.zeros((len(anno.get('name', [])),), dtype=np.float32)


def add_distance_eval(ap_result_str, ap_dict, eval_gt_annos, eval_det_annos, class_names, kitti_eval):
    ap_result_str += "\n\n" + "=" * 60 + "\n"
    ap_result_str += " Distance-based Evaluation (0-40 / 40-80 / 80-120 m) \n"
    ap_result_str += "=" * 60 + "\n"

    for min_dist, max_dist in DISTANCE_RANGES:
        cur_det_annos = []
        cur_gt_annos = []

        for det_anno, gt_anno in zip(eval_det_annos, eval_gt_annos):
            det = copy.deepcopy(det_anno)
            if len(det.get('name', [])) > 0:
                det_dist = _bev_distance(det)
                det_mask = (det_dist >= min_dist) & (det_dist < max_dist)
                det = _first_axis_filter(det, det_mask)
            cur_det_annos.append(det)

            gt = copy.deepcopy(gt_anno)
            if len(gt.get('name', [])) > 0:
                gt_dist = _bev_distance(gt)
                gt_mask = (gt_dist >= min_dist) & (gt_dist < max_dist)
                gt['name'] = np.where(gt_mask, gt['name'], 'DontCare')
            cur_gt_annos.append(gt)

        ap_result_str += f"\n--- Distance Range: {min_dist}m to {max_dist}m ---\n"
        dist_result_str, dist_dict = kitti_eval.get_official_eval_result(
            cur_gt_annos, cur_det_annos, class_names
        )
        ap_result_str += dist_result_str
        for key, value in dist_dict.items():
            ap_dict[f'Dist_{min_dist}_{max_dist}_{key}'] = value

    return ap_result_str, ap_dict
