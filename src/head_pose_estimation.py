"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""
import numpy as np
import os
import time
from openvino.inference_engine import IECore, IENetwork
import cv2


class HeadPoseEstimationModel:
    """
    Class for the Head Pose Estimation Model.
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
        preprocessing_start_time = time.time()
        preprocessed_image = self.preprocess_input(image)
        preprocessing_total_time = time.time() - preprocessing_start_time
        input_dict = {self.input_name: preprocessed_image}
        output = self.net.infer(input_dict)
        head_pose_angles = self.preprocess_output(output)

        return head_pose_angles, preprocessing_total_time

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

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        # output blob consists of three angles correspond to yaw, pitch and roll
        # each angle is a float array with shape [1, 1]
        yaw_angle = outputs["angle_y_fc"][0][0]
        pitch_angle = outputs["angle_p_fc"][0][0]
        roll_angle = outputs["angle_r_fc"][0][0]

        return [yaw_angle, pitch_angle, roll_angle]
