""" 
    Run this script to visualize predicted bounding boxes 
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from dataset import draw_bounding_boxes, YoloData
from metrics import YoloMetrics, YoloUtils
from model import YoloModel
import pyautogui, numpy, cv2 , os 
from typing import *

current_folder = os.path.dirname(__file__)

model_folder   = os.path.join(current_folder, "models/models")

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

def take_screenshot():
    image = cv2.cvtColor(numpy.uint8(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    image = image[5:605,5:805]
    return image 

if (__name__ == "__main__"):

    model_savename = os.path.join(model_folder, "yolo_v1p5-230321_230516.h5")

    # yolo constants / hyperparameters

    lambda_noobj = 0.5

    lambda_coord = 5.0 

    thresh_obj   = 0.20

    thresh_iou   = 0.10

    S            = 14 # do not change

    C            = 1

    input_shape  = (448, 448, 3) # do not change

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    yolo_model = YoloModel.load_model_from_disk(model_savename)
    
    yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision, optimizer = "adam", loss = "mse") 

    while True:

        images = numpy.stack([ cv2.resize(take_screenshot(), (input_shape[1], input_shape[0])) ])

        preprocessed_images = preprocess_input(numpy.stack([ cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images ]))

        predictions = yolo_model.predict(preprocessed_images)

        (pred_bounding_boxes) = (
            YoloUtils.convert_cells_to_bounding_boxes(predictions, S, C, True).numpy().tolist()
        )

        # pred_bounding_boxes : { (img_idx, obj, cls_idx, x, y, w, h) }
        pred_bounding_boxes = YoloUtils.non_max_suppression(pred_bounding_boxes, thresh_obj, thresh_iou)

        for idx, prediction in enumerate(predictions):

            image = images[idx]

            _bounding_boxes = list(filter(lambda x : x[0] == idx, pred_bounding_boxes))

            _bounding_boxes = list(map(lambda x : [ x[2], *x[-4:] ], _bounding_boxes))

            image = draw_bounding_boxes(image, _bounding_boxes)

            cv2.imshow("Wizard Detection Test", image) 

            cv2.waitKey(1)

    cv2.destroyAllWindows()