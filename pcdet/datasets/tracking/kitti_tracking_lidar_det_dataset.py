import copy
import pickle

import numpy as np
from skimage import io

from ...utils import box_utils, calibration_kitti, common_utils
from ..dataset import DatasetTemplate


class KittiTrackingLidarDetDataset(DatasetTemplate):
    """
    Detector training dataset for KITTI-tracking-style sequences whose GT boxes are
    already stored in lidar coordinates as annos['gt_boxes_lidar'].
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.det_infos = []
        self.include_data(self.mode)
        default_map = {name: name for name in class_names}
        self.map_class_to_kitti = self.dataset_cfg.get('MAP_CLASS_TO_KITTI', default_map)
        self.class_name_remap = self.dataset_cfg.get('CLASS_NAME_REMAP', {})
        self.default_class_name = self.dataset_cfg.get('DEFAULT_CLASS_NAME', None)

    def _remap_names(self, names):
        if len(names) == 0:
            return names
        return np.asarray([
            self.class_name_remap.get(name, self.default_class_name if self.default_class_name is not None else name)
            for name in names
        ])

    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading lidar detector infos')

        infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_file = self.root_path / info_path
            if not info_file.exists():
                continue
            with open(info_file, 'rb') as f:
                infos.extend(pickle.load(f))

        self.det_infos.extend(infos)

        if self.logger is not None:
            self.logger.info('Total frames for lidar detector dataset: %d', len(infos))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.det_infos) * self.total_epochs
        return len(self.det_infos)

    def get_lidar(self, info):
        lidar_path = self.root_path / info['point_cloud']['lidar_path']
        assert lidar_path.exists(), f'Missing lidar file: {lidar_path}'
        return np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, info['point_cloud'].get('num_features', 4))

    def get_image(self, info):
        image_path = self.root_path / info['image']['image_path']
        assert image_path.exists(), f'Missing image file: {image_path}'
        image = io.imread(image_path).astype(np.float32)
        return image / 255.0

    def get_calib(self, info):
        calib_dict = info['calib']
        return calibration_kitti.Calibration(calib_dict)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            return {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'alpha': np.zeros(num_samples),
            }

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            if 'calib' in batch_dict:
                calib = batch_dict['calib'][batch_index]
                image_shape = None
                if 'image_shape' in batch_dict:
                    cur_shape = batch_dict['image_shape'][batch_index]
                    image_shape = cur_shape.cpu().numpy() if hasattr(cur_shape, 'cpu') else cur_shape
                pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes[:, :7], calib)
                pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    pred_boxes_camera, calib, image_shape=image_shape
                )
                pred_dict['bbox'] = pred_boxes_img
                pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
                pred_dict['location'] = pred_boxes_camera[:, 0:3]
                pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
                pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)
        return annos

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.det_infos)

        info = copy.deepcopy(self.det_infos[index])
        points = self.get_lidar(info)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        input_dict = {
            'frame_id': f"{info.get('sequence_id', 'seq')}_{info.get('frame_id', index)}",
            'points': points,
        }

        if 'sequence_id' in info:
            input_dict['sequence_id'] = str(info['sequence_id'])
        if 'frame_idx' in info:
            input_dict['frame_idx'] = np.int32(info['frame_idx'])

        if 'images' in get_item_list or 'calib_matrices' in get_item_list or 'calib_matricies' in get_item_list:
            calib = self.get_calib(info)
            input_dict['calib'] = calib

            if 'images' in get_item_list:
                input_dict['images'] = self.get_image(info)
                input_dict['image_shape'] = np.array(info['image']['image_shape'], dtype=np.int32)

            if 'calib_matrices' in get_item_list or 'calib_matricies' in get_item_list:
                p2 = np.eye(4, dtype=np.float32)
                p2[:3, :4] = calib.P2

                r0 = np.eye(4, dtype=np.float32)
                r0[:3, :3] = calib.R0

                v2c = np.eye(4, dtype=np.float32)
                v2c[:3, :4] = calib.V2C

                input_dict['trans_lidar_to_img'] = p2 @ r0 @ v2c
                input_dict['trans_lidar_to_cam'] = v2c
                input_dict['trans_cam_to_img'] = p2

        if 'annos' in info:
            annos = common_utils.drop_info_with_name(copy.deepcopy(info['annos']), name='DontCare')
            annos['name'] = self._remap_names(annos['name'])
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': np.asarray(annos['gt_boxes_lidar'], dtype=np.float32).reshape(-1, 7),
            })

            if 'bbox' in annos and 'gt_boxes2d' in get_item_list:
                input_dict['gt_boxes2d'] = annos['bbox']

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if not self.det_infos or 'annos' not in self.det_infos[0]:
            return 'No ground-truth boxes for evaluation', {}

        from ..kitti import kitti_utils
        from ..kitti.det2d_eval_utils import eval_det2d_map50, has_valid_det2d
        from ..kitti.distance_eval_utils import add_distance_eval
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.det_infos]
        for anno in eval_gt_annos:
            anno['name'] = self._remap_names(anno['name'])

        if has_valid_det2d(eval_det_annos):
            det2d_gt_annos = copy.deepcopy(eval_gt_annos)
            det2d_result_str, det2d_dict = eval_det2d_map50(det2d_gt_annos, eval_det_annos, class_names)
        else:
            det2d_result_str, det2d_dict = '', {}

        kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=self.map_class_to_kitti)
        kitti_utils.transform_annotations_to_kitti_format(eval_gt_annos, map_name_to_kitti=self.map_class_to_kitti)

        kitti_class_names = [self.map_class_to_kitti[name] for name in class_names]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        if det2d_result_str:
            ap_result_str += '\n' + det2d_result_str
            ap_dict.update(det2d_dict)
        ap_result_str, ap_dict = add_distance_eval(
            ap_result_str=ap_result_str,
            ap_dict=ap_dict,
            eval_gt_annos=eval_gt_annos,
            eval_det_annos=eval_det_annos,
            class_names=kitti_class_names,
            kitti_eval=kitti_eval,
        )
        return ap_result_str, ap_dict
