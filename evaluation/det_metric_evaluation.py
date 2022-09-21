# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import uuid
import shutil

from pathlib import Path
from ppocr.metrics.eval_det_iou import DetectionIoUEvaluator
from ppocr.utils.e2e_metric.polygon_fast import iou
from tools.infer.utility import get_rotate_crop_image


class DetMetricEvaluation(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        if 'iou_values' in kwargs.keys():
            self.iou_values = sorted(kwargs['iou_values'])
        else:
            self.iou_values = None

        self.save_res_path = kwargs['save_res_path']
        self.save_dir = os.path.dirname(self.save_res_path)
        self.save_instances_dir = os.path.join(os.path.dirname(self.save_dir), 'instances_images')
        self.save_true_dir = os.path.join(os.path.dirname(self.save_dir), 'true_images')
        self.save_false_dir = os.path.join(os.path.dirname(self.save_dir), 'false_images')

        if os.path.isdir(self.save_true_dir):
            shutil.rmtree(self.save_true_dir)
        Path(self.save_true_dir).mkdir(parents=True, exist_ok=True)

        if os.path.isdir(self.save_false_dir):
            shutil.rmtree(self.save_false_dir)
        Path(self.save_false_dir).mkdir(parents=True, exist_ok=True)

        if os.path.isdir(self.save_instances_dir):
            shutil.rmtree(self.save_instances_dir)
        Path(self.save_instances_dir).mkdir(parents=True, exist_ok=True)

        self.f_out = open(os.path.join(self.save_instances_dir, 'test.txt'), 'w')
        if not os.path.exists(os.path.dirname(self.save_res_path)):
            os.makedirs(self.save_dir)

        self.evaluator = DetectionIoUEvaluator(iou_values=self.iou_values)
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, *args, **kwargs):
        '''
       batch: a list produced by dataloaders.
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2)
           polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
       preds: a list of dict produced by post process
            points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
       '''
        file_name = args[0][0]
        if isinstance(args[1], np.ndarray):
            img = args[1]
        else:
            img_dir = args[1]
            img = cv2.imread(os.path.join(img_dir, file_name))
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]
        texts = sum(batch[4], [])
        kie_classes = sum(batch[5], [])
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred['points']]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)
            self.draw_and_save_result(img, result, file_name, texts, kie_classes)

    def draw_and_save_result(self, img, result, file_name, texts, kie_classes):
        draw_img = np.copy(img)
        result = result[str(self.iou_values[0])]
        gt_polygon = result['gtPolPoints']
        pred_polygon = result['detPolPoints']
        pairs = result['pairs']
        iou_mat = result['iouMat']
        recall = result['recall']

        # Draw ground truth
        blue = (255, 0, 0)
        self.draw_gt_boxes(draw_img, gt_polygon, blue)

        # Draw predict
        red = (0, 0, 255)
        green = (0, 255, 0)
        tp_box = [pair['det'] for pair in pairs]
        tp_gt_box = [pair['gt'] for pair in pairs]
        texts = [texts[i] for i in tp_gt_box]
        kie_classes = [kie_classes[i] for i in tp_gt_box]
        iou_box = [round(iou_mat[pair['gt']][pair['det']], 2) for pair in pairs]
        for i, box in enumerate(pred_polygon):
            points = np.asarray(box, dtype=np.float32)
            box = np.asarray(box, dtype=np.int32).reshape((-1, 1, 2))

            if i in tp_box:
                crop_img = get_rotate_crop_image(img, points)

                gt_index = tp_box.index(i)

                kie_class = kie_classes[gt_index]
                text = texts[gt_index]

                base_name = os.path.basename(file_name)
                name, ext = os.path.splitext(base_name)
                save_name = "_".join([kie_class, name, str(uuid.uuid4())]) + ext

                assert cv2.imwrite(os.path.join(self.save_instances_dir, save_name), crop_img)
                self.f_out.write(f"{save_name}\t{text}\n")

                cv2.polylines(draw_img, [box], True, color=green, thickness=1)
                cv2.putText(draw_img, str(iou_box[tp_box.index(i)]), tuple(box[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            green, 1, cv2.LINE_AA)
            else:
                cv2.polylines(draw_img, [box], True, color=red, thickness=1)

        if recall == 1:
            assert cv2.imwrite(os.path.join(self.save_true_dir, os.path.basename(file_name)), draw_img)
        else:
            assert cv2.imwrite(os.path.join(self.save_false_dir, os.path.basename(file_name)), draw_img)

        # assert cv2.imwrite(os.path.join(self.save_dir, os.path.basename(file_name)), draw_img)

    def draw_pred_boxes(img, gt_polygon, ):
        pass

    def draw_gt_boxes(self, img, boxes, color):
        if len(boxes) > 0:
            for box in boxes:
                box = np.asarray(box, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [box], True, color=color, thickness=1)
        return img

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        if self.iou_values is None:
            metrics = self.evaluator.combine_results(self.results)
        else:
            metrics = self.evaluator.combine_results_multi_iou(self.results, self.iou_values)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results
