# yolo training
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam 
from dataset import YoloData, YoloDataAugmenter
from metrics import YoloMetrics
from model import YoloModel
import datetime, requests, os, gc

current_folder = os.path.dirname(__file__)

model_folder   = os.path.join(current_folder, "models/models")

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

def confirm_existent_model_training(model_name : str) -> int:
    """
        This function returns 1 if continuing training is prohibited.
    """

    # allow training to proceced since model name is not taken 
    if not (os.path.isfile(model_name)):
        return 0
    
    print("The specified model name already exists. Do you want to proceed anyways?")

    while True:

        confirm_overwrite = input("(YES/NO)> ")

        # overwrite the previous model 
        if (confirm_overwrite == "YES"):
            return 0
        
        # terminate the training procedure
        if (confirm_overwrite == "NO"):
            return 1 

model_savename = os.path.join(model_folder, "yolo_v1p5-{}.h5".format(datetime.datetime.now().strftime("%y%m%d_%H%M%S")))

#model_savename = os.path.join(model_folder, "yolo_v1p5-230327_215610.h5")

if (__name__ == "__main__"):

    # dataset directories

    dataset_version       = 5

    image_folder          = os.path.join(current_folder, f"../wiz-detect.v{dataset_version}i.yolov5pytorch/train/images")

    label_folder          = os.path.join(current_folder, f"../wiz-detect.v{dataset_version}i.yolov5pytorch/train/labels")

    val_image_folder      = os.path.join(current_folder, f"../wiz-detect.v{dataset_version}i.yolov5pytorch/valid/images")

    val_label_folder      = os.path.join(current_folder, f"../wiz-detect.v{dataset_version}i.yolov5pytorch/valid/labels")

    pretrain_image_folder = os.path.join(current_folder, f"../voc-dataset.v1i.yolov5pytorch/train/images")

    pretrain_label_folder = os.path.join(current_folder, "../voc-dataset.v1i.yolov5pytorch/train/labels")

    # yolo constants / hyperparameters

    lambda_noobj = 0.5

    lambda_coord = 5.0

    thresh_obj   = 0.05

    thresh_iou   = 0.20

    S            = 14 # do not change

    C            = 1

    pretrain_C   = 20

    input_shape  = (448, 448, 3) # do not change

    # model hyperparameters

    batch_size      = 24

    pretrain_epochs = 8

    epochs          = 128

    num_images      = 2048

    learning_rate   = 4e-5 #8e-5

    training_board  = "https://api.thingspeak.com/update?api_key=3I9WY6ON41AUZ6IL&field1={}&field2={}"

    pretrain_with_voc = False

    augment_list    = [] #[  YoloDataAugmenter.AUG_BRIGHT, YoloDataAugmenter.AUG_LRFLIP  ]

    if (pretrain_with_voc):

        if (confirm_existent_model_training(model_savename)):
            raise IOError("The specified model already exists. Training process is terminated.\n")

        # training on PASCAL VOC DATASET

        yolo_metrics = YoloMetrics(S, pretrain_C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

        yolo_model = YoloModel(C = pretrain_C)

        yolo_model.summary()

        yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision,
            optimizer = Adam(learning_rate = learning_rate), loss = yolo_metrics.loss) 

        yolo_data = YoloData(S, pretrain_C, input_shape)

        (quantity, generator) = yolo_data.initialize_generator(batch_size, pretrain_image_folder, pretrain_label_folder, num_images, augment_list = augment_list) 

        (val_quantity, val_generator) = yolo_data.initialize_generator(batch_size, pretrain_image_folder, pretrain_label_folder, num_images, use_augment = False, reverse_data = True)

        clear_session()
        gc.collect()

        for epoch in range(pretrain_epochs):

            print("Epoch: {}/{}".format(epoch + 1, pretrain_epochs))

            (images, labels) = next(generator())
            mean_AP = yolo_model.evaluate(images, labels, verbose = 1)

            (images, labels) = next(val_generator())
            val_mAP = yolo_model.evaluate(images, labels, verbose = 1)

            if (training_board is not None):
                try:
                    requests.get(training_board.format(mean_AP, val_mAP), timeout = 10)
                except:
                    pass 

            yolo_model.fit(generator(), batch_size = batch_size, epochs = 1, shuffle = True, steps_per_epoch = quantity)

            clear_session()
            gc.collect()

            yolo_model.save_middle_weights(model_savename)

        # fine-tuning on custom dataset

        if (training_board is not None):
            try:
                requests.get(training_board.format(-1, -1), timeout = 10)
            except:
                pass 

    yolo_metrics = YoloMetrics(S, C, lambda_coord, lambda_noobj, thresh_obj, thresh_iou)

    if (pretrain_with_voc):

        yolo_model = YoloModel(C = C)

        yolo_model.summary()

        yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision,
            optimizer = Adam(learning_rate = learning_rate), loss = yolo_metrics.loss) 
        
        yolo_model.load_middle_weights(model_savename)

    else:

        yolo_model = YoloModel.load_model_from_disk(model_savename)

        yolo_model.summary()

        yolo_model.compile(mean_average_precision = yolo_metrics.mean_average_precision,
            optimizer = Adam(learning_rate = learning_rate), loss = yolo_metrics.loss) 

    yolo_data = YoloData(S, C, input_shape)

    (quantity, generator) = yolo_data.initialize_generator(batch_size, image_folder, label_folder, augment_list = augment_list) 

    (val_quantity, val_generator) = yolo_data.initialize_generator(batch_size, val_image_folder, val_label_folder, use_augment = False)

    clear_session()
    gc.collect()

    for epoch in range(epochs):

        print("Epoch: {}/{}".format(epoch + 1, epochs))

        (images, labels) = next(generator())
        mean_AP = yolo_model.evaluate(images, labels, verbose = 1)

        (images, labels) = next(val_generator())
        val_mAP = yolo_model.evaluate(images, labels, verbose = 1)

        if (training_board is not None):
            try:
                requests.get(training_board.format(mean_AP, val_mAP), timeout = 10)
            except:
                pass 

        yolo_model.fit(generator(), batch_size = batch_size, epochs = 1, shuffle = True, steps_per_epoch = quantity)

        clear_session()
        gc.collect()

        yolo_model.save_model_to_disk(model_savename)