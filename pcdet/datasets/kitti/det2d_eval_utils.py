import numpy as np


def _bbox_iou(box, boxes):
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    area_a = np.maximum(box[2] - box[0], 0.0) * np.maximum(box[3] - box[1], 0.0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def _ap_from_pr(recalls, precisions):
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]) * 100.0)


def _valid_bbox_mask(bboxes):
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0,), dtype=bool)
    bboxes = np.asarray(bboxes)
    return (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])


def has_valid_det2d(det_annos):
    for anno in det_annos:
        bboxes = anno.get('bbox', None)
        if bboxes is not None and _valid_bbox_mask(bboxes).any():
            return True
    return False


def eval_det2d_map50(gt_annos, det_annos, class_names, iou_thresh=0.5):
    result_lines = ['2D Detection AP@0.50:']
    result_dict = {}
    class_aps = []

    for cls_name in class_names:
        gt_by_frame = {}
        npos = 0
        det_records = []

        for frame_idx, gt_anno in enumerate(gt_annos):
            names = np.asarray(gt_anno.get('name', []))
            bboxes = np.asarray(gt_anno.get('bbox', np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
            cls_mask = (names == cls_name) & _valid_bbox_mask(bboxes)
            cls_boxes = bboxes[cls_mask]
            gt_by_frame[frame_idx] = {
                'boxes': cls_boxes,
                'used': np.zeros((cls_boxes.shape[0],), dtype=bool),
            }
            npos += cls_boxes.shape[0]

        for frame_idx, det_anno in enumerate(det_annos):
            names = np.asarray(det_anno.get('name', []))
            bboxes = np.asarray(det_anno.get('bbox', np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
            scores = np.asarray(det_anno.get('score', np.zeros((len(names),), dtype=np.float32)), dtype=np.float32)
            if bboxes.shape[0] != names.shape[0]:
                continue
            cls_mask = (names == cls_name) & _valid_bbox_mask(bboxes)
            for bbox, score in zip(bboxes[cls_mask], scores[cls_mask]):
                det_records.append((float(score), frame_idx, bbox))

        if npos == 0:
            ap = None
        elif len(det_records) == 0:
            ap = 0.0
        else:
            det_records.sort(key=lambda item: item[0], reverse=True)
            tp = np.zeros((len(det_records),), dtype=np.float32)
            fp = np.zeros((len(det_records),), dtype=np.float32)

            for det_idx, (_, frame_idx, bbox) in enumerate(det_records):
                gt_info = gt_by_frame[frame_idx]
                gt_boxes = gt_info['boxes']
                ious = _bbox_iou(bbox, gt_boxes)
                best_idx = int(ious.argmax()) if ious.size > 0 else -1
                best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
                if best_iou >= iou_thresh and not gt_info['used'][best_idx]:
                    tp[det_idx] = 1.0
                    gt_info['used'][best_idx] = True
                else:
                    fp[det_idx] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / max(float(npos), 1.0)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
            ap = _ap_from_pr(recalls, precisions)

        if ap is not None:
            class_aps.append(ap)
            result_dict[f'det2d_{cls_name.lower()}_ap_0.5'] = ap
            result_lines.append(f'{cls_name} 2D AP@0.50: {ap:.4f}')
        else:
            result_lines.append(f'{cls_name} 2D AP@0.50: N/A')

    mean_ap = float(sum(class_aps) / len(class_aps)) if class_aps else None
    if mean_ap is not None:
        result_dict['det2d_map_0.5'] = mean_ap
        result_lines.append(f'2D mAP@0.50: {mean_ap:.4f}')
    else:
        result_lines.append('2D mAP@0.50: N/A')

    return '\n'.join(result_lines) + '\n', result_dict
