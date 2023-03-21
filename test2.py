""" 
    Run this script to visualize predicted bounding boxes 
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
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

    model_savename = os.path.join(model_folder, "yolo_v1p5-230319_104650.h5")

    test_folder = os.path.join(current_folder, "../test_images")

    image_names = YoloData.find_files_in_folder(test_folder)

    # yolo constants / hyperparameters

    lambda_noobj = 0.5

    lambda_coord = 5.0

    thresh_obj   = 0.05

    thresh_iou   = 0.30

    S            = 14 # do not change

    C            = 1

    input_shape  = (448, 448, 3) # do not change

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    yolo_model = YoloModel.load_model_from_disk(model_savename)
    
    yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision, optimizer = "adam", loss = "mse") 

    images = preprocess_input(numpy.stack([ cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), (input_shape[1], input_shape[0])) for filename in image_names ]))

    predictions = yolo_model.predict(images)

    (pred_bounding_boxes) = (
        YoloUtils.convert_cells_to_bounding_boxes(predictions, S, C,  True).numpy().tolist()
    )

    # pred_bounding_boxes : { (img_idx, obj, cls_idx, x, y, w, h) }
    pred_bounding_boxes = YoloUtils.non_max_suppression(pred_bounding_boxes, thresh_obj, thresh_iou)

    images = numpy.stack([ cv2.resize(cv2.imread(filename), (input_shape[1], input_shape[0])) for filename in image_names ])

    for idx, prediction in enumerate(predictions):

        image = images[idx]

        _bounding_boxes = list(filter(lambda x : x[0] == idx, pred_bounding_boxes))

        _bounding_boxes = list(map(lambda x : [ x[2], *x[-4:] ], _bounding_boxes))

        image = draw_bounding_boxes(image, _bounding_boxes)

        cv2.imshow("Wizard Detection Test", image) 

        cv2.waitKey(0)

        cv2.destroyAllWindows()