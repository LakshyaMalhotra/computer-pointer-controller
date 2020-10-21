"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import numpy as np
import time
from openvino.inference_engine import IECore, IENetwork
import cv2


class FacialLandmarksModel:
    """
    Class for the Facial landmarks detection model.
    """

    def __init__(self, model_name, device="CPU", extensions=None):
        """
        TODO: Use this to set your instance variables.
        """
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_weights = model_name + ".bin"
        self.model_structure = model_name + ".xml"

        try:
            self.core = IECore()
            try:
                self.model = self.core.read_network(
                    self.model_structure, self.model_weights
                )
            except AttributeError:
                self.model = IENetwork(
                    model=self.model_structure, weights=self.model_weights
                )
        except Exception:
            raise ValueError(
                "Could not initialize the network. Have you entered the correct model path?"
            )
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        """
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        """
        self.net = self.core.load_network(
            self.model, device_name=self.device, num_requests=1
        )

    def predict(self, image):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        # recording the time taken to preprocess the inputs
        preprocessing_start_time = time.time()
        preprocessed_image = self.preprocess_input(image)
        input_dict = {self.input_name: preprocessed_image}
        preprocessing_total_time = time.time() - preprocessing_start_time
        output = self.net.infer(input_dict)
        eyes_coordinates = self.preprocess_output(output, image)

        # extract x,y coordinates of both eyes' centers
        left_eye_x_coord = eyes_coordinates[0]
        left_eye_y_coord = eyes_coordinates[1]
        right_eye_x_coord = eyes_coordinates[2]
        right_eye_y_coord = eyes_coordinates[3]

        # finding the image of the eyes; trying with 5, might have to change later
        left_eye_x_min = left_eye_x_coord - 15
        left_eye_x_max = left_eye_x_coord + 15
        left_eye_y_min = left_eye_y_coord - 15
        left_eye_y_max = left_eye_y_coord + 15

        right_eye_x_min = right_eye_x_coord - 15
        right_eye_x_max = right_eye_x_coord + 15
        right_eye_y_min = right_eye_y_coord - 15
        right_eye_y_max = right_eye_y_coord + 15

        # cropping the image of left and right eye from the actual face image
        # left_eye_image = image[
        #     left_eye_x_min:left_eye_x_max, left_eye_y_min:left_eye_y_max
        # ]
        # right_eye_image = image[
        #     right_eye_x_min:right_eye_x_max, right_eye_y_min:right_eye_y_max
        # ]
        left_eye_image = image[
            left_eye_y_min:left_eye_y_max, left_eye_x_min:left_eye_x_max
        ]
        right_eye_image = image[
            right_eye_y_min:right_eye_y_max, right_eye_x_min:right_eye_x_max
        ]

        # storing the eye coordinates for both eyes
        left_eye_coords = [
            left_eye_x_min,
            left_eye_y_min,
            left_eye_x_max,
            left_eye_y_max,
        ]
        right_eye_coords = [
            right_eye_x_min,
            right_eye_y_min,
            right_eye_x_max,
            right_eye_y_max,
        ]

        # returning eye coordinates to show the bboxes on eyes
        eye_coords = [left_eye_coords, right_eye_coords]

        return (
            left_eye_image,
            right_eye_image,
            eye_coords,
            preprocessing_total_time,
        )

    def check_model(self):
        # checking for unsupported layers
        supported_layers = self.core.query_network(
            network=self.model, device_name=self.device
        )
        unsupported_layers = [
            layer
            for layer in self.model.layers.keys()
            if layer not in supported_layers
        ]
        # add device extension if unsupported layers are found
        if len(unsupported_layers) != 0:
            print("You have unsupported layers in your network...")
            try:
                print(
                    "You are using latest version of OpenVINO, don't need extensions"
                )
            except:
                self.core.add_extension(self.extensions, self.device)
                print("Extension is added to suppport the layers")

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        preprocessed_image = cv2.resize(
            image, (self.input_shape[3], self.input_shape[2])
        )
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        preprocessed_image = preprocessed_image.reshape(
            1, *preprocessed_image.shape
        )

        return preprocessed_image

    def preprocess_output(self, outputs, image):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        # model outputs a blob with shape [1,10] containing the x,y cooridnates of
        # two eyes, nose and two lip corners. All the coordinates are normalized
        # so we also need to scale them appropriately. We need the coordinates of the eyes.
        output = outputs[self.output_name][0]
        left_eye_x_coord = int(output[0] * image.shape[1])
        left_eye_y_coord = int(output[1] * image.shape[0])
        right_eye_x_coord = int(output[2] * image.shape[1])
        right_eye_y_coord = int(output[3] * image.shape[0])

        return [
            left_eye_x_coord,
            left_eye_y_coord,
            right_eye_x_coord,
            right_eye_y_coord,
        ]

