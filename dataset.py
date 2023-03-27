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

    def _randomly_focus_on_images(self, images : numpy.ndarray, labels : List[ numpy.ndarray ], min_scale : Optional[ float ] = 0.75) -> Tuple[ numpy.ndarray, List[ numpy.ndarray ] ]:
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
        
        # whether to zoom in (0) or out (1)
        IN_OUT = numpy.random.randint(0, 2, (n,), dtype = numpy.bool8)

        # zoom-out ratio if zooming out 
        OUT_SZ = numpy.random.uniform(min_scale, 1.0, (n,)).astype(numpy.float32)

        for idx, image in enumerate(images):

            # (B, 5) { cls_idx, x, y, w, h }
            label = labels[idx] 

            # skip augmenting the image and label altogether if there are no bounding boxes 
            if (label is None):
                augmented_images.append(image)
                augmented_labels.append(label)
                continue 

            # zooming out 
            if (IN_OUT[idx]):

                # initialize a blank image of same size 
                temp_image = numpy.zeros_like(image, dtype = numpy.uint8)

                # image shape and zoom-out ratio
                (img_h, img_w, m) = (*image.shape[0:2], OUT_SZ[idx])

                # actual width and height after shrinking image 
                (dx, dy) = (int(m * img_w), int(m * img_h))

                # actual top-left position to embed shrunk image 
                (sx, sy) = (
                    numpy.random.randint(0, img_w - dx + 1),
                    numpy.random.randint(0, img_h - dy + 1)
                )

                # embed shrunk image into blank image 
                temp_image[sy : sy + dy, sx : sx + dx] = cv2.resize(image, (dx, dy))

                augmented_images.append(temp_image)

                label[..., 1:2] = m * label[..., 1:2] + sx / img_w 

                label[..., 2:3] = m * label[..., 2:3] + sy / img_h 

                label[..., 3:4] = m * label[..., 3:4]

                label[..., 4:5] = m * label[..., 4:5]

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

    def __init__(self, S                : int, 
                       C                : int, 
                       input_shape      : Tuple[ int, int, int ], 
                       brightness_range : Optional[ Tuple[ float, float ] ] = (0.5, 1.1)) -> None:
        
        assert isinstance(S,                int  )
        assert isinstance(C,                int  )
        assert isinstance(input_shape,      tuple)
        assert isinstance(brightness_range, tuple)

        (self.S, self.C, self.input_shape, self.brightness_range) = (
            S, C, input_shape, brightness_range
        )

    @staticmethod 
    def find_files_in_folder(folder_name          : str, 
                             sort_names           : Optional[ bool ]                      = True, 
                             filter_by_extensions : Optional[ Union[ str, List[ str ] ] ] = None) -> List[ str ]:

        assert isinstance(folder_name, str) 
        assert isinstance(sort_names, bool) 

        def __check_folder_exists(folder_name : str) -> str:

            # raise an exception if the specified directory could not be found; otherwise, return the folder name 
            if not (os.path.isdir(folder_name)):
                raise IOError(f"The specified directory does not exist: \"{folder_name}\"\n")
            return folder_name 
        
        # all file names in the specified directory 
        filenames = list(map(lambda x : os.path.join(folder_name, x), os.listdir(__check_folder_exists(folder_name))))

        if (sort_names):

            # sort file names in alphabetical order [ A => Z ]
            filenames.sort()

        if (filter_by_extensions is not None):

            # "filter_by_extensions" must be "str" or "List[ str ]"
            if not (isinstance(filter_by_extensions, str) or isinstance(filter_by_extensions, list)):
                raise Exception(f"Parameter \"filter_by_extensions\" must be \"{str}\" or \"{list}\", not \"{type(filter_by_extensions)}\"\n")
            
            # convert "str" to "List[ str * 1 ]"
            if (isinstance(filter_by_extensions, str)):
                filter_by_extensions = [ filter_by_extensions ]

            # convert file extension strings to lowercase 
            filter_by_extensions = list(map(str.lower, filter_by_extensions))

            # remove file names with wrong file extensions 
            filenames = list(filter(lambda x : os.path.splitext(x)[1].lower() in filter_by_extensions, filenames))

        return filenames 
    
    def load_labels(self, label_files : List[ str ]) -> List[ Union[ numpy.ndarray, None ] ]:
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

    def load_images(self, filenames : List[ str ]) -> numpy.ndarray:
        (h, w, *_) = self.input_shape 
        loaded_images = numpy.stack([
            cv2.resize(cv2.imread(filename), (w, h))
                for filename in filenames 
        ])
        return loaded_images
    
    def _preprocess_images(self, images : numpy.ndarray) -> numpy.ndarray:
        return preprocess_input(numpy.stack([ cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images ]))

    def initialize_generator(self, batch_size   :           int, 
                                   image_folder :           str, 
                                   label_folder :           str, 
                                   max_images   : Optional[ int  ] = None, 
                                   use_augment  : Optional[ bool ] = True,
                                   shuffle_data : Optional[ bool ] = True,
                                   reverse_data : Optional[ bool ] = False,
                                   augment_list : Optional[ list ] = [   ]) -> Tuple[ int, Iterator[ tuple ] ]:
        
        assert isinstance(batch_size,   int )
        assert isinstance(image_folder, str )
        assert isinstance(label_folder, str )
        assert isinstance(max_images,   int ) or max_images is None 
        assert isinstance(use_augment,  bool)
        assert isinstance(shuffle_data, bool)
        assert isinstance(reverse_data, bool)
        assert isinstance(augment_list, list)

        (image_filenames, label_filenames) = (
            self.find_files_in_folder(image_folder),
            self.find_files_in_folder(label_folder)
        )

        if (reverse_data):
            (image_filenames, label_filenames) = (
                list(reversed(image_filenames)),
                list(reversed(label_filenames))
            )

        num_images = len(image_filenames)

        if (num_images == 0):
            raise Exception(f"The specified directory \"{image_folder}\" contains no images.\n")
        
        if (num_images != len(label_filenames)):
            raise Exception("The number of images and labels do not match.\n")

        if (max_images is not None):
            num_images = min(max_images, num_images)
        
        if (shuffle_data):

            indices = numpy.arange(0, num_images).astype(numpy.int64)
            
            numpy.random.shuffle(indices)

            image_filenames = list(map(lambda x : image_filenames[x], indices))

            label_filenames = list(map(lambda x : label_filenames[x], indices)) 

        def generate(include_original : Optional[ bool ] = False) -> Iterator[ tuple ]:
            nonlocal augment_list
            """ 
                Parameters:
                    [ 1 ] include_original [ whether to return original (untampered) images for illustration ]
                Results:
                    [ 1 ] (preprocessed_images, preprocessed_labels) [ data augmentation could be toggled by the parameter "use_augment" ]
                    [ 2 ] (preprocessed_images, preprocessed_labels, original_images) [ data augmentation could be used; original (BGR 0-255) images are appended ]
            """

            if (use_augment):

                # initialize the data augmenter for YOLO training 
                data_augmenter = YoloDataAugmenter(self.brightness_range)

            while True:

                for start_idx in range(0, num_images, batch_size):

                    indexing = slice(start_idx, min(start_idx + batch_size, num_images))

                    # image and label filenames of current batch 
                    (__image_filenames, __label_filenames) = (
                        image_filenames[ indexing ],
                        label_filenames[ indexing ]
                    ) 

                    # load images (BGR, 0-255, resized) and labels of current batch 
                    (images, labels) = (
                        self.load_images(__image_filenames),
                        self.load_labels(__label_filenames)
                    )

                    # employ data augmentation 
                    if (use_augment):


                        # use all available augmentation techniques if unspecified 
                        (images, labels) = data_augmenter.augment_images(images, labels, augments = ((augment_list) if (augment_list.__len__()) else (None)))

                    # preprocess images (ImageNet) and labels (YOLO)
                    preprocessed_data = (
                        self._preprocess_images(images),
                        self._preprocess_labels(labels)
                    )

                    # append original images to returning tuple 
                    if (include_original):
                        preprocessed_data = (*preprocessed_data, images)

                    yield preprocessed_data

        
        return (num_images, generate)
        