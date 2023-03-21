""" 
    Run this script to annotate images in target folder 
"""

""" 
    Note that annotations are made in PyTorch YOLO v5 format 
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.backend import clear_session
import numpy, cv2, sys, os, gc 
from dataset import YoloData 
from utils import YoloUtils
from model import YoloModel
from typing import *

def load_preprocess_images(filenames : List[ str ]) -> numpy.ndarray: 
    return preprocess_input(numpy.stack([
        cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), (448, 448))
            for filename in filenames 
    ]))

# yaml format [ data.yaml ]
""" 
train: ./train/images

nc: 1
names: ['wizard']
"""

if (__name__ == "__main__"):

    # basic constants & hyperparameters 

    current_folder = os.path.dirname(__file__)

    model_name     = os.path.join(current_folder, "models/models/yolo_v1p5-230319_104650.h5")

    image_folder   = os.path.join(current_folder, "../data/train")

    S = 14 # do not change 

    C = 1  # do not change 

    thresh_iou = 0.30

    thresh_obj = 0.05 

    batch_size = 32

    only_process_new = True 

    # 

    yolo_model = YoloModel.load_model_from_disk(model_name)

    yolo_model.compile(None, optimizer = "adam", loss = "mse")

    filenames = YoloData.find_files_in_folder(image_folder) 

    filenames = list(filter(lambda x : not os.path.isdir(x), filenames))

    if (only_process_new):
        filenames = list(filter(lambda x : not os.path.isfile(os.path.splitext(x)[0] + ".txt"), filenames))

    num_images = len(filenames) 

    int_padding = len(str(num_images))

    for start_idx in range(0, num_images, batch_size):

        sys.stdout.write("\r[ {0:>{1:}}/{2:} ]".format(start_idx + 1, int_padding, num_images)) 
        sys.stdout.flush() 

        end_idx = min(num_images, start_idx + batch_size) 

        src_images = load_preprocess_images(filenames[ start_idx : end_idx ])

        gc.collect() 

        clear_session()

        predictions = yolo_model.predict(src_images, verbose = 0)

        bounding_boxes = YoloUtils.extract_and_format_bounding_boxes(
            predictions, S = S, C = C, thresh_obj = thresh_obj, thresh_iou = thresh_iou).tolist()

        for img_idx in range(len(src_images)):

            _bounding_boxes = list(filter(lambda x : x[0] == img_idx, bounding_boxes)) 

            annotation_filename = os.path.splitext(filenames[ start_idx + img_idx ])[0] + ".txt" 

            with open(annotation_filename, "w", encoding = "utf-8") as wf:

                last_idx = len(_bounding_boxes) - 1

                for box_idx, bounding_box in enumerate(_bounding_boxes):

                    wf.write("{0:} {1:} {2:} {3:} {4:}".format(
                          int(bounding_box[2]), 
                        float(bounding_box[3]),
                        float(bounding_box[4]),
                        float(bounding_box[5]),
                        float(bounding_box[6])
                    ))

                    if (box_idx != last_idx):
                        wf.write("\n")

        sys.stdout.write("\r[ {0:>{1:}}/{2:} ]".format(end_idx, int_padding, num_images)) 
        sys.stdout.flush() 

    print("\n")