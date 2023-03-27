""" 
    Run this script to visualize ground-truth and predicted bounding boxes 
"""

from dataset import draw_bounding_boxes, YoloData
from metrics import YoloMetrics, YoloUtils
from model import YoloModel
import numpy, cv2 , os 
from typing import *

current_folder = os.path.dirname(__file__)

model_folder   = os.path.join(current_folder, "models/models")

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

if (__name__ == "__main__"):

    # >>> filepath to model and data 

    model_savename = os.path.join(model_folder, "yolo_v1p5-230325_214343.h5")

    image_folder   = os.path.join(current_folder, "../wiz-detect.v5i.yolov5pytorch/valid/images")

    label_folder   = os.path.join(current_folder, "../wiz-detect.v5i.yolov5pytorch/valid/labels")

    # <<< filepath to model and data 


    # >>> YOLO hyper-parameters

    lambda_noobj = 0.5

    lambda_coord = 5.0

    thresh_obj   = 0.05

    thresh_iou   = 0.2

    S            = 14 # do not change

    C            = 1

    input_shape  = (448, 448, 3) # do not change

    # <<< YOLO hyper-parameters

    #

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    yolo_model = YoloModel.load_model_from_disk(model_savename)
    
    yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision, optimizer = "adam", loss = "mse") 

    yolo_data = YoloData(S, C, input_shape)

    # 

    (quantity, generator) = yolo_data.initialize_generator(32, image_folder, label_folder, use_augment = False) 

    (preprocessed_images, preprocessed_labels, images) = next(generator(include_original = True))

    yolo_model.evaluate(preprocessed_images, preprocessed_labels, verbose = 1)

    predictions = yolo_model.predict(preprocessed_images, verbose = 0)

    (true_bounding_boxes, pred_bounding_boxes) = (
        YoloUtils.convert_cells_to_bounding_boxes(preprocessed_labels, S, C, False).numpy().tolist(),
        YoloUtils.convert_cells_to_bounding_boxes(predictions,         S, C,  True).numpy()
    )

    # pred_bounding_boxes : { (img_idx, obj, cls_idx, x, y, w, h) }
    pred_bounding_boxes = YoloUtils.non_max_suppression(pred_bounding_boxes, thresh_obj, thresh_iou)

    if (isinstance(pred_bounding_boxes, numpy.ndarray)):
        pred_bounding_boxes = pred_bounding_boxes.tolist()

    true_bounding_boxes = sorted(filter(lambda x : x[1] >= thresh_obj, true_bounding_boxes), key = lambda x : x[1], reverse = True)

    for idx, prediction in enumerate(predictions):

        image = images[idx]

        # >>> drawing detected bounding boxes 

        _bounding_boxes = list(filter(lambda x : x[0] == idx, pred_bounding_boxes))

        _bounding_boxes = list(map(lambda x : [ x[2], *x[-4:] ], _bounding_boxes))

        image = draw_bounding_boxes(image, _bounding_boxes)

        # <<< drawing detected bounding boxes 

        # >>> drawing ground-truth bounding boxes (RED)

        _bounding_boxes = list(filter(lambda x : x[0] == idx, true_bounding_boxes))

        _bounding_boxes = list(map(lambda x : [ x[2], *x[-4:] ], _bounding_boxes))

        image = draw_bounding_boxes(image, _bounding_boxes, box_color = (0, 0, 255))

        # <<< drawing ground-truth bounding boxes (RED)

        cv2.imshow("Wizard Detector - Model Visualizer", image) 

        cv2.waitKey(0)

    cv2.destroyAllWindows()