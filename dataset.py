# YOLO data

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy, copy, cv2, os
from typing import *

class YoloDataAugmenter: 

    AUG_BRIGHT = 0

    AUG_LRFLIP = 1

    AUG_ZOOMIN = 2

    def __init__(self, brightness_range : Optional[ Tuple[ float, float ] ] = (1.0, 1.0)):
        assert isinstance(brightness_range, tuple)

        self.brightness_range = brightness_range 

    def _randomly_adjust_brightness_of_images(self, images : numpy.ndarray) -> numpy.ndarray:
        """
            Parameters:
                [ 1 ] images : numpy.ndarray (N, H, W, C) [ pixels must be in range of 0 and 255 ]
        """

        assert isinstance(images, numpy.ndarray)
        (n, h, w, c) = images.shape
        brightness_weights = numpy.random.uniform(self.brightness_range[0], self.brightness_range[1], (n, 1)).repeat(h * w * c, axis = 1).reshape((n, h, w, c))
        return numpy.uint8(numpy.clip(numpy.float32(images) * brightness_weights, 0.0, 255.0))

    def _randomly_flip_images_horizontally(self, images : numpy.ndarray, labels : List[ numpy.ndarray ]) -> Tuple[ numpy.ndarray, List[ numpy.ndarray ] ]:
        """
            Parameters:
                [ 1 ] images : numpy.ndarray (N, H, W, C) 
                [ 2 ] labels : list of numpy.ndarray (N, B, 5) { cls_idx, x, y, w, h } { int, float * 4 } [ coordinates must be in range of 0 and 1 ]
        """

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, list)
        (n, h, w, c) = images.shape
        flipping_conditions = numpy.random.randint(0, 2, (n, 1), dtype = numpy.bool8)
        __should_flip_image = flipping_conditions.repeat(h * w * c, axis = 1).reshape((n, h, w, c))
        images = numpy.where(__should_flip_image, numpy.flip(images, axis = 2), images)
        for idx, should_flip_label in enumerate(flipping_conditions):
            # skip flipping label if there are no bounding boxes 
            if (labels[idx] is None):
                continue 
            if (should_flip_label[0]):
                labels[idx][..., 1:2] = (1.0 - labels[idx][..., 1:2])
        return (images, labels)

    def _randomly_focus_on_images(self, images : numpy.ndarray, labels : List[ numpy.ndarray ]) -> Tuple[ numpy.ndarray, List[ numpy.ndarray ] ]:
        """
            Parameters:
                [ 1 ] images : numpy.ndarray (N, H, W, C)
                [ 2 ] labels : list of numpy.ndarray (N, B, 5) { cls_idx, x, y, w, h } { int, float * 4 } [ coordinates must be in range of 0 and 1 ]
        """

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, list)
        (n, h, w, c) = images.shape
        augmented_images = []
        augmented_labels = []
        for idx, image in enumerate(images):

            # (B, 5) { cls_idx, x, y, w, h }
            label = labels[idx] 

            # skip augmenting the image and label altogether if there are no bounding boxes 
            if (label is None):
                augmented_images.append(image)
                augmented_labels.append(label)
                continue 

            # (B, 4) { sx, ex, sy, ey } [ xxyy format ]
            coord = numpy.concatenate([
                label[..., 1:2] - label[..., 3:4] / 2,
                label[..., 1:2] + label[..., 3:4] / 2,
                label[..., 2:3] - label[..., 4:5] / 2,
                label[..., 2:3] + label[..., 4:5] / 2
            ], axis = 1) 

            # find the minimum and maximum of x and y across all bounding boxes 
            (min_x, min_y, max_x, max_y) = (
                numpy.min(coord[..., 0:2]), 
                numpy.min(coord[..., 2:4]),
                numpy.max(coord[..., 0:2]),
                numpy.max(coord[..., 2:4])
            )

            # randomly choose a region to crop without slicing existing bounding boxes 
            (sx01, sy01, ex01, ey01) = (
                numpy.random.uniform(0.0, min_x),
                numpy.random.uniform(0.0, min_y),
                numpy.random.uniform(max_x, 1.0),
                numpy.random.uniform(max_y, 1.0)
            )

            # obtain pixel coordinates of the region of interest 
            (sx, sy, ex, ey) = (
                int(sx01 * w),
                int(sy01 * h),
                int(ex01 * w),
                int(ey01 * h)
            )

            # calculate the zoom ratio across both axes (x-y)
            (dx, dy) = (
                ex01 - sx01,
                ey01 - sy01
            )

            # crop the image, resize to original size and save it
            augmented_images.append(cv2.resize(image[ sy : ey, sx : ex ], (w, h)))

            # { cls_idx, x, y, w, h }
            label[..., 1:2] = (label[..., 1:2] - sx01) / dx 
            label[..., 2:3] = (label[..., 2:3] - sy01) / dy
            label[..., 3:4] = (label[..., 3:4]       ) / dx
            label[..., 4:5] = (label[..., 4:5]       ) / dy

            augmented_labels.append(label)
            
        return (numpy.stack(augmented_images), augmented_labels) 

    def augment_images(self, images : numpy.ndarray, labels : List[ numpy.ndarray ], augments : Optional[ List[ int ] ] = None) -> Tuple[ numpy.ndarray, List[ numpy.ndarray] ]:
        """
            Parameters:
                [ 1 ] images : numpy.ndarray (N, H, W, C)
                [ 2 ] labels : list of numpy.ndarray (N, B, 5) { cls_idx, x, y, w, h } { int, float * 4 } [ coordinates must be in range of 0 and 1 ]
                [ 3 ] augments : list of augmentation operations
        """

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, list)

        if (augments is None):
            augments = [ self.AUG_BRIGHT, self.AUG_LRFLIP, self.AUG_ZOOMIN ]

        for augment in augments:
            if (augment == self.AUG_BRIGHT):
                images = self._randomly_adjust_brightness_of_images(images)
                continue 
            if (augment == self.AUG_LRFLIP):
                images, labels = self._randomly_flip_images_horizontally(images, labels) 
                continue 
            if (augment == self.AUG_ZOOMIN):
                images, labels = self._randomly_focus_on_images(images, labels) 
                continue 

        return (images, labels)

def draw_bounding_boxes(image : numpy.ndarray, bounding_boxes : numpy.ndarray, box_color : Optional[ Tuple[ int, int, int ] ] = (0, 255, 0)) -> numpy.ndarray:
    (h, w, c) = image.shape
    for bounding_box in bounding_boxes:
        (t, x01, y01, w01, h01) = bounding_box 
        (sx, sy, ex, ey) = (
            int(w * (x01 - w01 / 2)), int(h * (y01 - h01 / 2)), int(w * (x01 + w01 / 2)), int(h * (y01 + h01 / 2))
        )
        image = cv2.rectangle(image, (sx, sy), (ex, ey), box_color, 2)
    return image

class YoloData:
    def __init__(self, S : int, C : int, input_shape : Tuple[ int, int, int ], use_pretrain : bool = True, brightness_range : Optional[ Tuple[ float, float ] ] = (0.5, 1.1)) -> None:
        assert isinstance(S, int)
        assert isinstance(C, int)
        assert isinstance(input_shape, tuple)
        assert isinstance(use_pretrain, bool)
        assert isinstance(brightness_range, tuple)
        (self.S, self.C, self.input_shape, self.use_pretrain, self.brightness_range) = (S, C, input_shape, use_pretrain, brightness_range)

    @staticmethod 
    def find_files_in_folder(folder_name : str) -> List[ str ]:
        filenames = sorted(map(lambda x : os.path.join(folder_name, x), os.listdir(folder_name)))
        return filenames 
    
    def load_labels(self, label_files : List[ str ]) -> List[ numpy.ndarray ]:
        def load_label(filename : str) -> numpy.ndarray:
            loaded_data = filter("".__ne__, open(filename, "r").read().split("\n"))
            loaded_label = []
            for line in loaded_data:
                loaded_label.append(numpy.float32(list(map(float, filter("".__ne__, line.split(" "))))))
            return ((numpy.stack(loaded_label)) if (len(loaded_label)) else (None))
        return [  load_label(filename) for filename in label_files  ]
    
    def _preprocess_labels(self, labels : List[ numpy.ndarray ]) -> numpy.ndarray:
        """
            Parameters:
                [ 1 ] labels : list of numpy.ndarray (N, B, 5) [ [ { cls_idx, x, y, w, h } ] ]
            Returns:
                [ 1 ] numpy.ndarray (N, S, S, C + 5)
        """
        def preprocess_label(label_data : numpy.ndarray) -> numpy.ndarray:
            """
                Parameters:
                    [ 1 ] label_data : numpy.ndarray (B, 5) [ { cls_idx, x, y, w, h } ]
                Returns:
                    [ 1 ] label : numpy.ndarray (S, S, C + 5)
            """
            label = numpy.zeros(shape = (self.S, self.S, self.C + 5), dtype = numpy.float32)
            # return empty label matrix if no bounding boxes are found 
            if (label_data is None):
                return label 
            for line in label_data:
                if (len(line) == 0):
                    continue 
                (c, x, y, w, h) = (int(line[0]), *line[1:])
                (u_x, u_y) = (int(x * self.S), int(y * self.S))
                (n_x, n_y) = (min(1, max(0, x - u_x / self.S)) * self.S, min(1, max(0, y - u_y / self.S) * self.S))
                label[u_y, u_x, c] = 1.0
                label[u_y, u_x, self.C : self.C + 5] = numpy.float32([ 1.0, n_x, n_y, w, h ]) 
            return label 
        return numpy.stack([
            preprocess_label(label)
                for label in labels 
        ])

    def load_images(self, filenames : List[ str ], is_test_set : Optional[ bool ] = False) -> numpy.ndarray:
        (h, w, c) = self.input_shape 
        loaded_images = numpy.stack([
            cv2.cvtColor(cv2.resize(cv2.imread(filename), (w, h)), cv2.COLOR_BGR2RGB)
                for filename in filenames 
        ])
        if (is_test_set):
            return numpy.stack([
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                    for image in loaded_images 
            ])
        return loaded_images
    
    def _preprocess_images(self, images : numpy.ndarray, is_test_set : bool = False) -> numpy.ndarray:
        if (is_test_set):
            return images 
        return ((preprocess_input(images)) if (self.use_pretrain) else (images / 255.0))

    def initialize_generator(self, batch_size : int, image_folder : str, label_folder : str, max_images : Optional[ int ] = None) -> Tuple[ int, Iterator[ Tuple[ numpy.ndarray, numpy.ndarray ] ] ]:
        assert isinstance(batch_size, int)
        assert isinstance(image_folder, str)
        assert isinstance(label_folder, str)
        assert isinstance(max_images, int) or max_images is None 

        (image_filenames, label_filenames) = (
            self.find_files_in_folder(image_folder),
            self.find_files_in_folder(label_folder)
        )

        assert len(image_filenames)
        assert len(label_filenames)

        if (max_images is None):
            max_images = int(1e20)
            
        max_images = min((max_images, len(image_filenames), len(label_filenames))) 

        def generate(is_test_set : Optional[ bool ] = False):
            data_augmenter = YoloDataAugmenter(self.brightness_range)
            while True:
                for start_idx in range(0, max_images, batch_size):
                    end_idx = min(start_idx + batch_size, max_images)
                    (__image_filenames, __label_filenames) = (
                        image_filenames[start_idx:end_idx],
                        label_filenames[start_idx:end_idx]
                    ) 

                    (images, labels) = (
                        self.load_images(__image_filenames, is_test_set),
                        self.load_labels(__label_filenames)
                    )

                    if not (is_test_set):
                        (images, labels) = data_augmenter.augment_images(images, labels)#, augments = [ data_augmenter.AUG_BRIGHT, data_augmenter.AUG_LRFLIP ])

                    (images, labels) = (
                        self._preprocess_images(images, is_test_set),
                        self._preprocess_labels(labels)
                    )
                    yield (
                        images, labels
                    )
        
        return (max_images, generate)
        