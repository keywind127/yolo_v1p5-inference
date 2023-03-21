# yolo training
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam 
from metrics import YoloMetrics
from dataset import YoloData
from model import YoloModel
import datetime, requests, os, gc

current_folder = os.path.dirname(__file__)

model_folder   = os.path.join(current_folder, "models/models")

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_savename = os.path.join(model_folder, "yolo_v1p5-{}.h5".format(datetime.datetime.now().strftime("%y%m%d_%H%M%S")))

if (__name__ == "__main__"):

    # dataset directories

    """
    image_folder = os.path.join(current_folder, "train-wiz/images")

    label_folder = os.path.join(current_folder, "train-wiz/labels")

    pretrain_image_folder = os.path.join(current_folder, "train-voc/images")

    pretrain_label_folder = os.path.join(current_folder, "train-voc/labels")
    """

    image_folder          = os.path.join(current_folder, "../wiz-detect.v1i.yolov5pytorch/train/images")

    label_folder          = os.path.join(current_folder, "../wiz-detect.v1i.yolov5pytorch/train/labels")

    pretrain_image_folder = os.path.join(current_folder, "../voc-dataset.v1i.yolov5pytorch/train/images")

    pretrain_label_folder = os.path.join(current_folder, "../voc-dataset.v1i.yolov5pytorch/train/labels")

    # yolo constants / hyperparameters

    lambda_noobj = 0.5

    lambda_coord = 5.0

    thresh_obj   = 0.5

    thresh_iou   = 0.5

    S            = 14 # do not change

    C            = 1

    pretrain_C   = 20

    input_shape  = (448, 448, 3) # do not change

    # model hyperparameters

    batch_size      = 32

    pretrain_epochs = 10

    epochs          = 128

    num_images      = 2048 + 512

    learning_rate   = 4e-5 #8e-5

    training_board  = None #"https://api.thingspeak.com/update?api_key=D8J9XHHCIDTQCAF9&field1={}"

    # training on PASCAL VOC DATASET

    yolo_metrics = YoloMetrics(S, pretrain_C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    yolo_model = YoloModel(C = pretrain_C)

    yolo_model.summary()

    yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision,
        optimizer = Adam(learning_rate = learning_rate), loss = yolo_metrics.loss) 

    yolo_data = YoloData(S, pretrain_C, input_shape)

    (quantity, generator) = yolo_data.initialize_generator(batch_size, pretrain_image_folder, pretrain_label_folder, num_images) 

    clear_session()
    gc.collect()

    for epoch in range(pretrain_epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        (images, labels) = next(generator())
        mean_AP = yolo_model.evaluate(images, labels, verbose = 1)
        if (training_board is not None):
            try:
                requests.get(training_board.format(mean_AP), timeout = 10)
            except:
                pass 
        yolo_model.fit(generator(), batch_size = batch_size, epochs = 1, shuffle = True, steps_per_epoch = quantity)
        clear_session()
        gc.collect()

        yolo_model.save_middle_weights(model_savename)

    # fine-tuning on custom dataset

    if (training_board is not None):
        try:
            requests.get(training_board.format(-1), timeout = 10)
        except:
            pass 

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    yolo_model = YoloModel(C = C)

    yolo_model.summary()

    yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision,
        optimizer = Adam(learning_rate = learning_rate), loss = yolo_metrics.loss) 

    yolo_model.load_middle_weights(model_savename)

    yolo_data = YoloData(S, C, input_shape)

    (quantity, generator) = yolo_data.initialize_generator(batch_size, image_folder, label_folder, num_images) 

    clear_session()
    gc.collect()

    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        (images, labels) = next(generator())
        mean_AP = yolo_model.evaluate(images, labels, verbose = 1)
        if (training_board is not None):
            try:
                requests.get(training_board.format(mean_AP), timeout = 10)
            except:
                pass 
        yolo_model.fit(generator(), batch_size = batch_size, epochs = 1, shuffle = True, steps_per_epoch = quantity)
        clear_session()
        gc.collect()

        yolo_model.save_model_to_disk(model_savename)