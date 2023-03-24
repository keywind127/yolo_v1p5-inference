""" 
    Run this script to visualize predicted bounding boxes 
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from dataset import draw_bounding_boxes, YoloData
from metrics import YoloMetrics, YoloUtils
from wizard import WizardCounter
from model import YoloModel
import pyautogui, numpy, cv2 , os 
from typing import *

current_folder = os.path.dirname(__file__)

model_folder   = os.path.join(current_folder, "models/models")

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

def take_screenshot():
    global wizard_client_screenshot
    image = cv2.cvtColor(numpy.uint8(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    image = image[5:605,5:805]
    wizard_client_screenshot = image 
    return image 

wizard_client_screenshot = numpy.random.randint(0, 256, (600, 800, 3), dtype = numpy.uint8)

def _detect_wizards(source_images : numpy.ndarray) -> numpy.ndarray:
    return yolo_model.predict(source_images, verbose = 0)

def cell2box(prediction_output : numpy.ndarray) -> numpy.ndarray:
    return YoloUtils.convert_cells_to_bounding_boxes(prediction_output, S, C) 

def _preprocess_input(source_images : numpy.ndarray) -> numpy.ndarray:
    return preprocess_input(numpy.stack([ cv2.cvtColor(cv2.resize(image, (input_shape[1], input_shape[0])), cv2.COLOR_BGR2RGB) for image in source_images ]))

def identify_client_wizards(wizard_counter_history_size : int = 5) -> Iterator[ Tuple[ numpy.ndarray, int ] ]:
    wizard_counter_history = WizardCounter(wizard_counter_history_size)
    while True:
        bounding_boxes = YoloUtils.remove_small_bounding_boxes(YoloUtils.non_max_suppression(cell2box(_detect_wizards(_preprocess_input([ wizard_client_screenshot ])))))
        wizard_counter_history.add(((0) if (bounding_boxes is None) else (len(bounding_boxes))))
        yield (bounding_boxes, wizard_counter_history.freq_val)

if (__name__ == "__main__"):

    model_savename = os.path.join(model_folder, "yolo_v1p5-230321_230516.h5")

    # yolo constants / hyperparameters

    lambda_noobj = 0.5

    lambda_coord = 5.0 

    thresh_obj   = 0.05

    thresh_iou   = 0.10

    S            = 14 # do not change

    C            = 1

    input_shape  = (448, 448, 3) # do not change

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    yolo_model = YoloModel.load_model_from_disk(model_savename)
    
    yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision, optimizer = "adam", loss = "mse") 

    wizard_identifier = identify_client_wizards(10)

    while True:

        images = numpy.stack([ cv2.resize(take_screenshot(), (input_shape[1], input_shape[0])) ])

        pred_bounding_boxes, num_wizards = next(wizard_identifier) 

        print("Num Wizards: ", num_wizards)

        for idx, _ in enumerate(images):

            image = images[idx]

            _bounding_boxes = list(filter(lambda x : x[0] == idx, pred_bounding_boxes))

            _bounding_boxes = list(map(lambda x : [ x[2], *x[-4:] ], _bounding_boxes))

            image = draw_bounding_boxes(image, _bounding_boxes)

            cv2.imshow("Wizard Detection Test", image) 

            cv2.waitKey(1)

    cv2.destroyAllWindows()