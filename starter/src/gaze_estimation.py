"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import numpy as np
import time
from openvino.inference_engine import IECore, IENetwork
import cv2


class GazeEstimationModel:
    """
    Class for the Gaze Estimation Model.
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

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        """
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        """
        # recording the time taken to preprocess the inputs
        preprocessing_start_time = time.time()
        processed_left_eye_img = self.preprocess_input(left_eye_image)
        processed_right_eye_img = self.preprocess_input(right_eye_image)
        preprocessing_total_time = time.time() - preprocessing_start_time
        input_dict = {
            "left_eye_image": processed_left_eye_img,
            "right_eye_image": processed_right_eye_img,
            "head_pose_angles": head_pose_angles,  # obtained from head pose estimation model
        }
        output = self.net.infer(input_dict)
        x_y_coords = self.preprocess_output(
            output
        )  # returns normalized coordinates for mouse pointer

        return x_y_coords, preprocessing_total_time

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
        # model description:
        # https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
        try:
            preprocessed_image = cv2.resize(image, (60, 60))  # size:[1x3x60x60]
            preprocessed_image = preprocessed_image.transpose((2, 0, 1))
            preprocessed_image = preprocessed_image.reshape(
                1, *preprocessed_image.shape
            )
        except:
            print("Error while preprocessing the image")

        return preprocessed_image

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        # model description:
        # https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
        print(outputs)
        output = outputs[self.output_name][0]
        x = output[0]
        y = output[1]
        z = output[2]
        weight = 0.2  # coefficient to control the distance travelled by mouse pointer for better visulaization
        x = (x * weight) / cv2.norm(output)
        y = (y * weight) / cv2.norm(output)

        return [x, y, output]  # output is gaze vector
