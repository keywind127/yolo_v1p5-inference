# yolo loss

from tensorflow.python.framework.ops import EagerTensor 
from tensorflow.keras.backend import epsilon
from utils import YoloUtils
import tensorflow as tf 
from typing import *
import numpy 

class YoloMetrics(YoloUtils):

    def __init__(self, S : int, C : int, lambda_coord : float = 5.0, 
            lambda_noobj : float = 0.5, thresh_obj : float = 0.5, thresh_iou : float = 0.5) -> None:
        assert isinstance(S, int)
        assert isinstance(C, int)
        assert isinstance(lambda_coord, float)
        assert isinstance(lambda_noobj, float)
        assert isinstance(thresh_obj, float)
        assert isinstance(thresh_iou, float)
        (self.S, self.C, self.lambda_coord, self.lambda_noobj, self.thresh_obj, self.thresh_iou) = (
            S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou
        )

    def loss(self, y_true : EagerTensor, y_pred : EagerTensor) -> EagerTensor:
        """
            Parameters:
                [ 1 ] y_true : (N, S, S, C + 05)
                [ 2 ] y_pred : (N, S, S, C + 10)
            Restrictions:
                [ 1 ] all values are between 0.0 and 1.0
            Results:
                [ 1 ] total_loss : (,)
        """

        # all values must be in [ 0.0, 1.0 ]
        tf.debugging.assert_greater_equal(y_true, 0.0)
        tf.debugging.assert_less_equal(y_true, 1.0)
        tf.debugging.assert_greater_equal(y_pred, 0.0)
        tf.debugging.assert_less_equal(y_pred, 1.0)

        # mask indicating if cell contains objects : (N, S, S, 1)
        object_exists = y_true[..., self.C : self.C + 1]

        # no-objectness loss : (,)
        noobj_loss = self.lambda_noobj * tf.reduce_mean(
            tf.reduce_sum((1 - object_exists) * (
                tf.square(y_true[..., self.C : self.C + 1] - y_pred[..., self.C     : self.C + 1]) + 
                tf.square(y_true[..., self.C : self.C + 1] - y_pred[..., self.C + 5 : self.C + 6])
            ), axis = [1, 2, 3])
        )

        # classification loss : (,)
        class_loss = tf.reduce_mean(
            tf.reduce_sum(object_exists * tf.square(y_true[..., 0 : self.C] - y_pred[..., 0 : self.C]), axis = [1, 2, 3])
        )

        # bounding box scores : (N, S, S, 2)
        bounding_box_scores = tf.concat([
            self.intersection_over_union(y_true[..., self.C + 1 : self.C + 5], y_pred[..., self.C + 1 : self.C +  5]),
            self.intersection_over_union(y_true[..., self.C + 1 : self.C + 5], y_pred[..., self.C + 6 : self.C + 10])
        ], axis = 3)

        # only keeping bounding box info : (N, S, S, 5)
        y_true = y_true[..., self.C : self.C + 5]

        # only keeping the best bounding boxes : (N, S, S, 5)
        y_pred = self.reduce_select_bounding_boxes(y_pred[..., self.C : self.C + 10], bounding_box_scores)

        # coordination loss of xy : (,)
        xy_loss = self.lambda_coord * tf.reduce_mean(
            tf.reduce_sum(object_exists * tf.square(y_true[..., 1:3] - y_pred[..., 1:3]), axis = [1, 2, 3])
        )

        # coordination loss of wh : (,)
        wh_loss = self.lambda_coord * tf.reduce_mean(tf.reduce_sum(
            object_exists * tf.square(tf.sqrt(y_true[..., 3:5] + epsilon()) - 
                tf.sign(y_pred[..., 3:5]) * tf.sqrt(tf.abs(y_pred[..., 3:5]) + epsilon())
            ), axis = [1, 2, 3]
        ))

        # objectness loss : (,)
        obj_loss = tf.reduce_mean(
            tf.reduce_sum(object_exists * tf.square(y_true[..., 0:1] - y_pred[..., 0:1]), axis = [1, 2, 3])
        )

        # total loss : (,)
        total_loss = noobj_loss + obj_loss + xy_loss + wh_loss + class_loss 

        return total_loss 
    
    def mean_average_precision(self, y_true : EagerTensor, y_pred : EagerTensor) -> float:
        """
            Parameters:
                [ 1 ] y_true : (N, S, S, 25)
                [ 2 ] y_pred : (N, S, S, 30)
        """

        (true_bounding_boxes, pred_bounding_boxes) = (
            self.convert_cells_to_bounding_boxes(y_true, self.S, self.C, False).numpy().tolist(),
            self.convert_cells_to_bounding_boxes(y_pred, self.S, self.C,  True).numpy()
        )

        # pred_bounding_boxes : { (img_idx, obj, cls_idx, x, y, w, h) }
        pred_bounding_boxes = self.non_max_suppression(pred_bounding_boxes, self.thresh_obj, self.thresh_iou)

        if (isinstance(pred_bounding_boxes, numpy.ndarray)):
            pred_bounding_boxes = pred_bounding_boxes.tolist()

        true_bounding_boxes = sorted(filter(lambda x : x[1] >= self.thresh_obj, true_bounding_boxes), key = lambda x : x[1], reverse = True)

        ground_truth_history = [ 0 ] * len(true_bounding_boxes)

        # true_bounding_boxes : { (gnd_idx, img_idx, obj, cls_idx, x, y, w, h) }
        true_bounding_boxes = list(map(lambda x : [ x[0], *x[1] ], enumerate(true_bounding_boxes)))

        average_precisions = []

        for class_idx in range(self.C):

            (pred_boxes, true_boxes) = (
                list(filter(lambda x : x[2] == class_idx, pred_bounding_boxes)),
                list(filter(lambda x : x[3] == class_idx, true_bounding_boxes))
            )

            if (true_boxes.__len__() == 0):
                continue

            (true_positive, false_positive) = (
                numpy.zeros(len(pred_boxes)), numpy.zeros(len(pred_boxes))
            )

            for pred_idx, pred_box in enumerate(pred_boxes):

                __true_boxes = list(filter(lambda x : x[1] == pred_box[0], true_boxes))

                if not (__true_boxes):
                    false_positive[pred_idx] = 1
                    continue

                (best_idx, best_iou) = (0, -1)

                for true_box_idx, true_box in enumerate(__true_boxes):

                    temp_iou = self.list_intersection_over_union(true_box[-4:], pred_box[-4:])

                    if (best_iou < temp_iou):
                        (best_idx, best_iou) = (true_box_idx, temp_iou)

                if ((best_iou < self.thresh_iou) or (ground_truth_history[__true_boxes[best_idx][0]])):
                    false_positive[pred_idx] = 1
                    continue 

                true_positive[pred_idx] = 1
                ground_truth_history[__true_boxes[best_idx][0]] = 1

            (TP_cumsum, FP_cumsum) = (
                numpy.cumsum(true_positive, axis = 0),
                numpy.cumsum(false_positive, axis = 0)
            )
            (precisions, recalls) = (
                numpy.concatenate([ [ 1 ], TP_cumsum / (TP_cumsum + FP_cumsum    + epsilon()) ]),
                numpy.concatenate([ [ 0 ], TP_cumsum / (len(true_boxes) + epsilon()) ])
            )
            average_precisions.append(numpy.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)
