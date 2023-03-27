# yolo utilities 

from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.backend import epsilon
import tensorflow as tf 
from typing import *
import numpy

class YoloUtils:

    @classmethod 
    def intersection_over_union(class_, bounding_box_1 : EagerTensor, bounding_box_2 : EagerTensor) -> EagerTensor:
        """
            Parameters:
                [ 1 ] bounding_box_1 : (N, S, S, 4)
                [ 2 ] bounding_box_2 : (N, S, S, 4)
            Restrictions:
                [ 1 ] bounding box values are bounded between 0.0 and 1.0
                [ 2 ] bounding boxes are of type (x, y, w, h) where (x, y) is the center
            Results:
                [ 1 ] intersection_over_union : (N, S, S, 1)
        """

        # all values must be in [ 0.0, 1.0 ]
        tf.debugging.assert_greater_equal(bounding_box_1, 0.0)
        tf.debugging.assert_less_equal(bounding_box_1, 1.0)
        tf.debugging.assert_greater_equal(bounding_box_2, 0.0)
        tf.debugging.assert_less_equal(bounding_box_2, 1.0)

        # 1st bounding box area : (N, S, S, 1)
        box_area_1 = bounding_box_1[..., 2:3] * bounding_box_1[..., 3:4]

        # 2nd bounding box area : (N, S, S, 1)
        box_area_2 = bounding_box_2[..., 2:3] * bounding_box_2[..., 3:4]

        # intersection bounding box sx : (N, S, S, 1)
        min_x = tf.maximum(
            bounding_box_1[..., 0:1] - bounding_box_1[..., 2:3] / 2,
            bounding_box_2[..., 0:1] - bounding_box_2[..., 2:3] / 2
        )

        # intersection bounding box ex : (N, S, S, 1)
        max_x = tf.minimum(
            bounding_box_1[..., 0:1] + bounding_box_1[..., 2:3] / 2,
            bounding_box_2[..., 0:1] + bounding_box_2[..., 2:3] / 2
        )

        # intersection bounding box sy : (N, S, S, 1)
        min_y = tf.maximum(
            bounding_box_1[..., 1:2] - bounding_box_1[..., 3:4] / 2,
            bounding_box_2[..., 1:2] - bounding_box_2[..., 3:4] / 2
        )

        # intersection bounding box ey : (N, S, S, 1)
        max_y = tf.minimum(
            bounding_box_1[..., 1:2] + bounding_box_1[..., 3:4] / 2,
            bounding_box_2[..., 1:2] + bounding_box_2[..., 3:4] / 2
        )

        # intersection box area : (N, S, S, 1)
        intersection_area = tf.maximum(max_x - min_x, 0.0) * tf.maximum(max_y - min_y, 0.0)

        # intersection over union : (N, S, S, 1)
        intersection_over_union = (
            (intersection_area) / (box_area_1 + box_area_2 - intersection_area + epsilon())
        )

        return intersection_over_union 
    
    @staticmethod 
    def list_intersection_over_union(bounding_box_1 : List[ float ], bounding_box_2 : List[ float ]) -> float:
        """
            Limitations:
                [ 1 ] bounding_box_1 and bounding_box_2 must contain 4 values
        """

        assert len(bounding_box_1) == 4
        assert len(bounding_box_2) == 4

        (box_area_1, box_area_2) = (
            bounding_box_1[2] * bounding_box_1[3],
            bounding_box_2[2] * bounding_box_2[3]
        )

        (bounding_box_1, bounding_box_2) = ([ 
                bounding_box_1[0] - bounding_box_1[2] / 2,
                bounding_box_1[1] - bounding_box_1[3] / 2,
                bounding_box_1[0] + bounding_box_1[2] / 2,
                bounding_box_1[1] + bounding_box_1[3] / 2
            ], [ 
                bounding_box_2[0] - bounding_box_2[2] / 2,
                bounding_box_2[1] - bounding_box_2[3] / 2,
                bounding_box_2[0] + bounding_box_2[2] / 2,
                bounding_box_2[1] + bounding_box_2[3] / 2
            ]
        )

        (x_min, x_max, y_min, y_max) = (
            max(bounding_box_1[0], bounding_box_2[0]), 
            min(bounding_box_1[2], bounding_box_2[2]),
            max(bounding_box_1[1], bounding_box_2[1]), 
            min(bounding_box_1[3], bounding_box_2[3])
        )

        intersection = max(x_max - x_min, 0) * max(y_max - y_min, 0)

        return intersection / (box_area_1 + box_area_2 - intersection + epsilon())
    
    @classmethod 
    def reduce_select_bounding_boxes(class_, y_pred : EagerTensor, box_scores : EagerTensor) -> EagerTensor:
        """ 
            Parameters:
                [ 1 ] y_pred     : (N, S, S, 10)
                [ 2 ] box_scores : (N, S, S, 02)
            Results:
                [ 1 ] results : (N, S, S, 5)
        """

        # reshaping the bounding boxes for easier indexing : (N, S, S, 2, 5)
        y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], 2, 5))

        # axis-3 indices : (N, S, S)
        L = tf.argmax(box_scores, axis = 3)

        # axis-0, axis-1, axis-2 indices : (N, S, S)
        I, J, K = tf.meshgrid(tf.range(tf.shape(L)[0]), tf.range(tf.shape(L)[1]), tf.range(tf.shape(L)[2]), indexing = "ij")

        # bounding box indices : (N, S, S, 4)
        bounding_box_indices = tf.concat([
            tf.cast(tf.expand_dims(I, axis = 3), tf.int32),
            tf.cast(tf.expand_dims(J, axis = 3), tf.int32),
            tf.cast(tf.expand_dims(K, axis = 3), tf.int32),
            tf.cast(tf.expand_dims(L, axis = 3), tf.int32)
        ], axis = 3)

        # extracted bounding boxes : (N, S, S, 5)
        results = tf.gather_nd(y_pred, bounding_box_indices)

        return results
    
    @classmethod 
    def convert_cells_to_bounding_boxes(class_, label_data : EagerTensor, S : int, C : int, is_prediction : bool = True) -> EagerTensor:
        """
            Parameters:
                [ 1 ] label_data : (N, S, S, C + 10) or (N, S, S, C + 5) 
                [ 2 ] S { grid cell size }
                [ 3 ] is_prediction { whether if is prediction or label }
            Returns:
                [ 1 ] bounding_boxes : (N * S * S, 7) { (img_idx, obj, cls_idx, x, y, w, h) }
        """

        label_data = tf.cast(label_data, tf.float32)

        # label_data : (N, S, S, C + 10) or (N, S, S, C + 5) [ clip off values (x, y, w, h, c) out of Range[0, 1] ]
        label_data = tf.clip_by_value(label_data, 0, 1)

        # C : (N, S, S, 1) [ determine class for each grid cell ]
        T = tf.cast(tf.expand_dims(tf.argmax(label_data[..., 0 : C], axis = 3), axis = 3), tf.float32)

        # label_data : (N, S, S, 10) or (N, S, S, 5) [ remove class probabilities ]
        label_data = label_data[..., C : ]

        if (is_prediction):

            # bounding_box_scores : (N, S, S, 2)
            bounding_box_scores = tf.concat([
                label_data[..., 0:1], label_data[..., 5:6]
            ], axis = 3)

            # label_data : (N, S, S, 5) [ extract only the best bounding boxes ]
            label_data = class_.reduce_select_bounding_boxes(label_data, bounding_box_scores)

        # I, J, K : (N, S, S) [ respectively (img_idx, row_idx, col_idx) ]
        (I, J, K) = tf.meshgrid(tf.range(tf.shape(label_data)[0]), tf.range(S), tf.range(S), indexing = "ij")

        # I, J, K : (N, S, S, 1) [ expand the number of dimensions, and cast the indices as 32-bit floats ]
        (I, J, K) = (
            tf.cast(tf.expand_dims(I, axis = 3), tf.float32),
            tf.cast(tf.expand_dims(J, axis = 3), tf.float32),
            tf.cast(tf.expand_dims(K, axis = 3), tf.float32)
        )

        # X, Y, W, H, O : (N, S, S, 1) [ calculate x-y relative to the entire image and extract w-h-o sub-tensors ]
        (X, Y, W, H, O) = (
            (label_data[..., 1:2] + K) / S,
            (label_data[..., 2:3] + J) / S,
            (label_data[..., 3:4]    )    ,
            (label_data[..., 4:5]    )    ,
            (label_data[..., 0:1]    )
        )

        # bounding_boxes : (N * S * S, 7) [ assemble bounding box info (img_idx, obj, cls_idx, x, y, w, h) and reshape the tensor ] 
        bounding_boxes = tf.reshape(tf.concat([  I, O, T, X, Y, W, H  ], axis = 3), (-1, 7))

        return bounding_boxes 

    @classmethod 
    def non_max_suppression(class_, bounding_boxes : numpy.ndarray, thresh_obj : float = 0.5, thresh_iou : float = 0.5) -> Union[ numpy.ndarray, list ]:
        """ 
            Parameters:
                [ 1 ] bounding_boxes : (N * S * S * B, 7) { img_idx, obj_idx, cls_idx, x, y, w, h }
                [ 2 ] thresh_obj [ minimum objectness score ]
                [ 3 ] thresh_iou [ maximum overlap score ]   
            Results:
                [ 1 ] "None" if there are no bounding boxes after NMS
                [ 2 ] suppressed_bounding_boxes : (-1, 7) otherwise      
        """

        # force casts bounding boxes to NumPy array
        if not isinstance(bounding_boxes, numpy.ndarray):
            bounding_boxes = numpy.float32(bounding_boxes)

        # bounding_boxes : (-1, 7) [ remove bounding boxes with objectness score lower than threshold ]
        bounding_boxes = numpy.reshape(bounding_boxes[numpy.where(numpy.reshape(numpy.repeat(bounding_boxes[..., 1:2] >= thresh_obj, 7), (-1, 7)))], (-1, 7))

        # return empty list if there are no remaining bounding boxes 
        if (numpy.shape(bounding_boxes)[0] == 0):
            return []

        # bounding_boxes : (-1, 7) [ sort bounding boxes by objectness score in non-ascending order ]
        bounding_boxes = bounding_boxes[numpy.argsort(bounding_boxes[..., 1])][::-1]

        def intersection_over_union(bounding_boxes : numpy.ndarray, selected_bounding_box : numpy.ndarray, epsilon : Optional[ float ] = 1e-9) -> numpy.ndarray:
            """
                Parameters:
                    [ 1 ] bounding_boxes        : (-1, 4)
                    [ 2 ] selected_bounding_box : ( 1, 4) 
            """

            # selected_bounding_box : (-1, 4) [ reshape selected bounding box to simplify matrix operations ]
            selected_bounding_box = numpy.repeat(selected_bounding_box, numpy.shape(bounding_boxes)[0], axis = 0)

            (x_min, x_max, y_min, y_max) = (
                numpy.maximum(bounding_boxes[..., 0:1] - bounding_boxes[..., 2:3] / 2, selected_bounding_box[..., 0:1] - selected_bounding_box[..., 2:3] / 2),
                numpy.minimum(bounding_boxes[..., 0:1] + bounding_boxes[..., 2:3] / 2, selected_bounding_box[..., 0:1] + selected_bounding_box[..., 2:3] / 2),
                numpy.maximum(bounding_boxes[..., 1:2] - bounding_boxes[..., 3:4] / 2, selected_bounding_box[..., 1:2] - selected_bounding_box[..., 3:4] / 2),
                numpy.minimum(bounding_boxes[..., 1:2] + bounding_boxes[..., 3:4] / 2, selected_bounding_box[..., 1:2] + selected_bounding_box[..., 3:4] / 2)
            )

            # intersection : (-1, 1)
            intersection = numpy.maximum(x_max - x_min, 0) * numpy.maximum(y_max - y_min, 0)

            # union : (-1, 1)
            union = bounding_boxes[..., 2:3] * bounding_boxes[..., 3:4] + selected_bounding_box[..., 2:3] * selected_bounding_box[..., 3:4] - intersection 

            return intersection / (union + epsilon)

        def suppress_remaining_boxes(bounding_boxes : numpy.ndarray, selected_bounding_box : numpy.ndarray) -> numpy.ndarray:
            """ 
                Parameters:
                    [ 1 ] bounding_boxes        : (-1, 7)
                    [ 2 ] selected_bounding_box : ( 1, 7)
            """

            # different_image : (-1, 1) [ whether bounding boxes are in other images ]
            different_image = bounding_boxes[..., 0:1] != selected_bounding_box[0, 0]

            # different_class : (-1, 1) [ whether bounding boxes are in other classes ]
            different_class = bounding_boxes[..., 2:3] != selected_bounding_box[0, 2]

            # should_keep     : (-1, 1) [ whether bounding boxes have different "img_idx" or "cls_idx" from selected box ]
            should_keep = numpy.bitwise_or(different_image, different_class)

            # should_not_keep : (-1, 1) [ opposite of "should_keep" ]
            should_not_keep = numpy.bitwise_not(should_keep) 

            # low_iou : (-1, 1) [ whether bounding boxes have low "IOU" with selected box, False by default ]
            low_iou = numpy.zeros(shape = (numpy.shape(bounding_boxes)[0], 1), dtype = numpy.bool8) 

            if (numpy.any(should_not_keep)):

                # bounding_box_info : (-1, 4) [ bounding boxes (xywh) sharing "img_idx" and "cls_idx" with selected box ]
                bounding_box_info = numpy.reshape(bounding_boxes[numpy.where(numpy.reshape(numpy.repeat(should_not_keep, 7), (-1, 7)))], (-1, 7))[..., -4:]

                low_iou[ numpy.where(should_not_keep) ] = (intersection_over_union(bounding_box_info, selected_bounding_box[..., -4:]) <= thresh_iou).ravel()

            # should_keep : (-1, 7) [ whether to keep bounding box info ]
            should_keep = numpy.reshape(numpy.repeat(numpy.bitwise_or(should_keep, low_iou), 7, axis = 1), (-1, 7)) 

            # bounding_boxes : (-1, 7) [ keep bounding boxes with different "img_idx" or "cls_idx" or low "IOU" with selected box ]
            bounding_boxes = numpy.reshape(bounding_boxes[numpy.where(should_keep)], (-1, 7))

            return bounding_boxes 

        suppressed_bounding_boxes = []

        while (bounding_boxes.__len__()):

            suppressed_bounding_boxes.append(bounding_boxes[0]) 

            bounding_boxes = suppress_remaining_boxes(bounding_boxes[1:], bounding_boxes[0:1]) 

        return numpy.stack(suppressed_bounding_boxes)
    
    @classmethod 
    def remove_small_bounding_boxes(class_, bounding_boxes : numpy.ndarray, min_area : Optional[ float ] = 0.005) -> Union[ numpy.ndarray, list ]:
        """
            Parameters:
                [ 1 ] bounding_boxes : (-1, 7) { img_idx, obj_idx, cls_idx, x, y, w, h }
                [ 2 ] min_area [ minimum area for each bounding box ] 
            Results:
                [ 1 ] bounding_boxes : (-1, 7) 
        """

        # return empty list if there are no bounding boxes 
        if (bounding_boxes.__len__() == 0):
            return []

        # force casts bounding boxes to NumPy array
        if not isinstance(bounding_boxes, numpy.ndarray):
            bounding_boxes = numpy.float32(bounding_boxes)

        # should_keep : (-1, 7)
        should_keep = numpy.reshape(numpy.repeat((bounding_boxes[..., 5:6] * bounding_boxes[..., 6:7]) >= min_area, 7), (-1, 7))

        if not (numpy.any(should_keep)):
            return []

        # bounding_boxes : (-1, 7) [ bounding boxes with areas larger than threshold ]
        bounding_boxes = numpy.reshape(bounding_boxes[ numpy.where(should_keep) ], (-1, 7))

        return bounding_boxes 

    @classmethod 
    def extract_and_format_bounding_boxes(class_, pred_boxes : EagerTensor, S : int, C : int, thresh_obj : float = 0.5, thresh_iou : float = 0.5) -> numpy.ndarray:
        """
            Parameters:
                [ 1 ] pred_boxes : (N, S, S, C + 10)
                [ 2 ] S
                [ 3 ] C
                [ 4 ] thresh_obj 
                [ 5 ] thresh_iou
        """

        # extracted_bounding_boxes : (N * S * S, 7)
        extracted_bounding_boxes = class_.convert_cells_to_bounding_boxes(pred_boxes, S, C) 

        # convert to list of list [ [ img_idx, obj_scr, cls_idx, x, y, w, h ] ]
        extracted_bounding_boxes = extracted_bounding_boxes.numpy()#.tolist()

        # suppress non-maximal bounding boxes 
        extracted_bounding_boxes = class_.non_max_suppression(extracted_bounding_boxes, thresh_obj, thresh_iou)

        # sort bounding boxes by image index
        #extracted_bounding_boxes.sort(key = lambda x : x[0])

        # convert to numpy array : (N * S * S, 7)
        #extracted_bounding_boxes = numpy.float32(extracted_bounding_boxes)

        return extracted_bounding_boxes 