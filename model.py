# yolo model 

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import (
    BatchNormalization,
    AveragePooling2D,
    Concatenate,
    LeakyReLU,
    Softmax,
    Conv2D,
    Lambda,
    Input
)

from typing import *
import pickle

class YoloModel(Model):

    def __init__(self, C : int, input_shape : Tuple[ int, int, int ], *args, **kwargs) -> None:
        assert isinstance(C, int)
        assert isinstance(input_shape, tuple)
        super(YoloModel, self).__init__(*args, **kwargs)
        (self.C, self._input_shape) = (
            C, input_shape 
        )
        self.input_layer = Input(self._input_shape)
        self.convolution = MobileNetV2(weights = "imagenet", include_top = False, input_tensor = self.input_layer)
        for layer in self.convolution.layers:
            layer.trainable = False 

        self.sub_model = Sequential([
            Conv2D(1024,        (1, 1), name = "yolo_conv_1", padding = "same"),
            LeakyReLU(0.1,              name = "yolo_relu_1"),
            BatchNormalization(         name = "yolo_norm_1"),
            Conv2D(1536,        (5, 5), name = "yolo_conv_2", padding = "same"),
            LeakyReLU(0.1,              name = "yolo_relu_2"),
            BatchNormalization(         name = "yolo_norm_2"),
            #AveragePooling2D(           name = "yolo_pool_2"),
            Conv2D(1280,        (1, 1), name = "yolo_conv_3", padding = "same"),
            LeakyReLU(0.1,              name = "yolo_relu_3"),
            BatchNormalization(         name = "yolo_norm_3")
        ])
        """
        self.input_layer_2 = Input(shape = self.convolution.output_shape)
        #
        self.conv_1 = Conv2D(1024,        (1, 1), name = "yolo_conv_1", padding = "same")(self.input_layer_2)
        self.relu_1 = LeakyReLU(0.1,              name = "yolo_relu_1")(self.conv_1)
        self.norm_1 = BatchNormalization(         name = "yolo_norm_1")(self.relu_1)
        #
        self.conv_2 = Conv2D(1536,        (5, 5), name = "yolo_conv_2", padding = "same")(self.norm_1)
        self.relu_2 = LeakyReLU(0.1,              name = "yolo_relu_2")(self.conv_2)
        self.norm_2 = BatchNormalization(         name = "yolo_norm_2")(self.relu_2)
        #
        self.pool_2 = AveragePooling2D(           name = "yolo_pool_2")(self.norm_2)
        #
        self.conv_3 = Conv2D(1280,        (1, 1), name = "yolo_conv_3", padding = "same")(self.pool_2)
        self.relu_3 = LeakyReLU(0.1,              name = "yolo_relu_3")(self.conv_3)
        self.norm_3 = BatchNormalization(         name = "yolo_norm_3")(self.relu_3)

        self.sub_model = Model(self.input_layer_2, self.norm_3)
        #
        """
        self.conv_4 = Conv2D(self.C + 10, (3, 3), name = "yolo_conv_4", padding = "same", activation = "sigmoid")
        #

        self.out = self.call(self.input_layer)
        self.build((None, *self._input_shape))

    def save_middle_weights(self, filename : str) -> None:
        with open(filename, "wb") as wf:
            pickle.dump(self.sub_model.get_weights(), wf)

    def load_middle_weights(self, filename : str) -> None:
        with open(filename, "rb") as rf:
            self.sub_model.set_weights(pickle.load(rf))

    def call(self, inputs : EagerTensor) -> EagerTensor:
        x = self.convolution(inputs)
        x = self.sub_model(x)
        """
        #x = self.norm_1(self.relu_1(self.conv_1(x)))
        #x = self.norm_2(self.relu_2(self.conv_2(x)))
        #x = self.pool_2(x)
        #x = self.norm_3(self.relu_3(self.conv_3(x)))
        """
        x = self.conv_4(x)
        return x 
    
    def compile(self, mean_average_precision : Callable, *args, **kwargs) -> None:
        self.mean_average_precision = mean_average_precision 
        super(YoloModel, self).compile(*args, **kwargs)
        
    def evaluate(self, test_data : EagerTensor, test_labl : EagerTensor, verbose : int = 0) -> float:
        result = self.predict(test_data)
        mean_average_precision = self.mean_average_precision(test_labl, result)
        if (verbose):
            print("mAP: {}".format(mean_average_precision))
        return mean_average_precision

    def save_as_functional_model(self, filename : str, *args, **kwargs) -> None:
        output_layer = self.convolution.output 
        output_layer = self.sub_model(output_layer)
        output_layer = self.conv_4(output_layer)
        Model(self.input_layer, output_layer).save(filename, *args, **kwargs)

if (__name__ == "__main__"):

    test = 0

    if (test == 1):
            
        yolo_model = YoloModel(20, (448, 448, 3))

        yolo_model.compile(None, optimizer = "adam", loss = "mse")

        import tensorflow as tf 

        yolo_model.save_as_functional_model("models/models/model_save.h5")#, include_optimizer = False)

    if (test == 2):

        from tensorflow.keras.models import load_model 

        yolo_model = load_model("models/models/model_save.h5")

        yolo_model.summary() 