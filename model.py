# yolo model 

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.layers import (
    BatchNormalization,
    LeakyReLU,
    Conv2D,
    Input
)
from typing import *
import pickle, json, copy, os

class YoloModel(Model):

    default_architecture_config = {
        "num_classes" : 20,
        "feature_extractor" : "mobilenet_v2",
        "prediction_layers" : [
            [ "conv2d", 1024, [ 1, 1 ], "same", "yolo_conv_1" ],
            [ "leakyrelu", "yolo_relu_1" ],
            [ "batchnorm", "yolo_norm_1" ],
            [ "conv2d", 1536, [ 5, 5 ], "same", "yolo_conv_2" ],
            [ "leakyrelu", "yolo_relu_2" ],
            [ "batchnorm", "yolo_norm_2" ],
            [ "conv2d", 1280, [ 1, 1 ], "same", "yolo_conv_3" ],
            [ "leakyrelu", "yolo_relu_3" ],
            [ "batchnorm", "yolo_norm_3" ]
        ],
        "output_name" : "yolo_conv_f",
        "output_conv" : [ 3, 3 ]
    }

    def __init__(self, architecture_config : Optional[ Dict[ str, Union[ int, str, list ] ] ] = None, C : Optional[ int ] = None, *args, **kwargs) -> None:
        """
            Configurations:
                [ 1 ] input shape : (-1, 448, 448, 3)
                [ 2 ] grid cells  : 14 x 14 
        """
        super(YoloModel, self).__init__(*args, **kwargs)
        self.architecture_config = copy.deepcopy(
            (self.default_architecture_config) 
                if (architecture_config is None) else (architecture_config)
        )
        if (C is not None):
            assert isinstance(C, int)
            self.architecture_config["num_classes"] = C
        self.input_layer = Input((448, 448, 3))
        assert self.architecture_config["feature_extractor"] == "mobilenet_v2"
        self.convolution = MobileNetV2(weights = "imagenet", include_top = False, input_tensor = self.input_layer)
        for layer in self.convolution.layers:
            layer.trainable = False 
        self.sub_model = Sequential()
        for layer_info in self.architecture_config["prediction_layers"]:
            (layer_type, *layer_details, layer_name) = layer_info 
            if (layer_type == "conv2d"):
                self.sub_model.add(Conv2D(layer_details[0], tuple(layer_details[1]), padding = layer_details[2], name = layer_name))
                continue 
            if (layer_type == "leakyrelu"):
                self.sub_model.add(LeakyReLU(0.1, name = layer_name))
                continue 
            if (layer_type == "batchnorm"):
                self.sub_model.add(BatchNormalization(name = layer_name))
                continue 

        self.conv_f = Conv2D(self.architecture_config["num_classes"] + 10, tuple(self.architecture_config["output_conv"]), 
            padding = "same", name = self.architecture_config["output_name"], activation = "sigmoid"
        )

        self.out = self.call(self.input_layer)
        self.build((None, 448, 448, 3))
        self.mean_average_precision = None 

    def save_middle_weights(self, filename : str) -> None:
        with open(filename, "wb") as wf:
            pickle.dump(self.sub_model.get_weights(), wf)

    def load_middle_weights(self, filename : str) -> None:
        with open(filename, "rb") as rf:
            self.sub_model.set_weights(pickle.load(rf))

    def save_model_to_disk(self, filename : str, *args, **kwargs) -> None:
        if os.path.splitext(filename)[1].lower() not in [ ".h5", ".h5py" ]:
            raise Exception(f"The specified file format is not supported: \"{filename}\"\n")
        config_filename = os.path.splitext(filename)[0] + ".json"
        self.save_weights(filename, *args, **kwargs)
        open(config_filename, "w", encoding = "utf-8").write(json.dumps(self.architecture_config, indent = 4))

    @classmethod 
    def load_model_from_disk(class_, filename : str, *args, **kwargs) -> "YoloModel":
        assert os.path.splitext(filename)[1].lower() in [ ".h5", ".h5py" ] 
        config_filename = os.path.splitext(filename)[0] + ".json"
        if not os.path.isfile(filename):
            raise IOError(f"The specified file could not be found: \"{filename}\"\n")
        if not os.path.isfile(config_filename):
            raise IOError(f"The specified file could not be found: \"{config_filename}\"\n")
        yolo_model = class_(json.load(open(config_filename, "r", encoding = "utf-8")))
        yolo_model.load_weights(filename, *args, **kwargs)
        return yolo_model 

    def call(self, inputs : EagerTensor) -> EagerTensor:
        x = self.convolution(inputs)
        x = self.sub_model(x)
        x = self.conv_f(x)
        return x 

    def compile(self, mean_average_precision : Optional[ Callable ] = None, *args, **kwargs) -> None:
        self.mean_average_precision = mean_average_precision 
        super(YoloModel, self).compile(*args, **kwargs)

    def evaluate(self, test_data : EagerTensor, test_labl : EagerTensor, verbose : int = 0) -> float:
        if (self.mean_average_precision is None):
            raise Exception("You must first compile the model with a \"mean average precision\" function.\n")
        result = self.predict(test_data, verbose = 0)
        mean_average_precision = self.mean_average_precision(test_labl, result)
        if (verbose):
            print("mAP: {}".format(mean_average_precision))
        return mean_average_precision

    def save_as_functional_model(self, filename : str, *args, **kwargs) -> None:
        output_layer = self.convolution.output 
        output_layer = self.sub_model(output_layer)
        output_layer = self.conv_f(output_layer)
        Model(self.input_layer, output_layer).save(filename, *args, **kwargs)

if (__name__ == "__main__"):

    test = 0

    if (test == 1):

        test_save_filename = "models/models/test_model_save.h5"

        # create default yolo model for 20 classes 
        yolo_model = YoloModel(C = 20)

        # save weights of prediction layers 
        yolo_model.save_middle_weights(test_save_filename)

        # load weights of prediction layers 
        yolo_model.load_middle_weights(test_save_filename)

        # save the weights (.h5) and architecture (.json)
        yolo_model.save_model_to_disk(test_save_filename)

        # load the model from weights (.h5) and architecture (.json)
        yolo_model = YoloModel.load_model_from_disk(test_save_filename)

        # compile the model (first param is mean_average_precision function)
        yolo_model.compile(None, optimizer = "adam", loss = "mse")

        yolo_model.summary()

        # save the model as a stand-alone keras model 
        yolo_model.save_as_functional_model(test_save_filename, include_optimizer = False)

        # load the model as a stand-alone keras model 
        yolo_model = load_model(test_save_filename)

        yolo_model.compile(optimizer = "adam", loss = "mse")

        yolo_model.summary()