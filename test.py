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

    model_savename = os.path.join(model_folder, "yolo_v1_vgg19-230317_014910.h5")

    image_folder          = os.path.join(current_folder, "../Wizard-Detection.v1i.yolov5pytorch/train/images")

    label_folder          = os.path.join(current_folder, "../Wizard-Detection.v1i.yolov5pytorch/train/labels")

    pretrain_image_folder = os.path.join(current_folder, "../voc-dataset.v1i.yolov5pytorch/train/images")

    pretrain_label_folder = os.path.join(current_folder, "../voc-dataset.v1i.yolov5pytorch/train/labels")

    # yolo constants / hyperparameters

    lambda_noobj = 0.5

    lambda_coord = 5.0

    thresh_obj   = 0.2

    thresh_iou   = 0.2

    S            = 7 # do not change

    C            = 1

    pretrain_C   = 20

    input_shape  = (448, 448, 3) # do not change

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    yolo_model = YoloModel(C, input_shape)

    yolo_model.load_weights(model_savename)
    
    yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision, optimizer = "adam", loss = "mse") 

    yolo_data = YoloData(S, C, input_shape)

    (quantity, generator) = yolo_data.initialize_generator(32, image_folder, label_folder) 

    (images, labels) = next(generator())

    yolo_model.evaluate(images, labels, verbose = 1)

    predictions = yolo_model.predict(images)

    (images, labels) = next(generator(is_test_set = True))

    #bounding_boxes = YoloUtils.extract_and_format_bounding_boxes(predictions, S, C, thresh_obj, thresh_iou).tolist()

    (true_bounding_boxes, pred_bounding_boxes) = (
        YoloUtils.convert_cells_to_bounding_boxes(labels, S, C, False).numpy().tolist(),
        YoloUtils.convert_cells_to_bounding_boxes(predictions, S, C,  True).numpy().tolist()
    )

    # pred_bounding_boxes : { (img_idx, obj, cls_idx, x, y, w, h) }
    pred_bounding_boxes = YoloUtils.non_max_suppression(pred_bounding_boxes, thresh_obj, thresh_iou)

    true_bounding_boxes = sorted(filter(lambda x : x[1] >= thresh_obj, true_bounding_boxes), key = lambda x : x[1], reverse = True)

    for idx, prediction in enumerate(predictions):

        image = images[idx]

        _bounding_boxes = list(filter(lambda x : x[0] == idx, pred_bounding_boxes))

        _bounding_boxes = list(map(lambda x : [ x[2], *x[-4:] ], _bounding_boxes))

        image = draw_bounding_boxes(image, _bounding_boxes)

        _bounding_boxes = list(filter(lambda x : x[0] == idx, true_bounding_boxes))

        _bounding_boxes = list(map(lambda x : [ x[2], *x[-4:] ], _bounding_boxes))

        image = draw_bounding_boxes(image, _bounding_boxes, box_color = (0, 0, 255))

        cv2.imshow("Wizard Detection Test", image) 

        cv2.waitKey(0)

        cv2.destroyAllWindows()