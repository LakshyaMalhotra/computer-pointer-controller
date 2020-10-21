"""
Actual python script to run the application.
"""
import sys
import os
import time
import cv2
import numpy as np
import logging as log

from argparse import ArgumentParser

# model scripts
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel
from mouse_controller import MouseController
from input_feeder import InputFeeder


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser(
        description="Run the application on demo video", allow_abbrev=True
    )
    # arguments for the paths to various models
    parser.add_argument(
        "-fd",
        "--face_detection",
        required=True,
        type=str,
        help="Path to the xml file of the trained face-detection model.",
    )
    parser.add_argument(
        "-fld",
        "--facial_landmarks",
        required=True,
        type=str,
        help="Path to the xml file of the trained facial landmarks detection model.",
    )
    parser.add_argument(
        "-hpe",
        "--head_pose",
        required=True,
        type=str,
        help="Path to the xml file of the trained head pose estimation model",
    )
    parser.add_argument(
        "-ge",
        "--gaze",
        required=True,
        type=str,
        help="Path to the xml file of the trained gaze estimation model",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        type=str,
        help="Path to input image or video file or webcam (CAM)",
    )
    parser.add_argument(
        "-l",
        "--cpu_extension",
        required=False,
        type=str,
        default=None,
        help="MKLDNN (CPU)-targeted custom layers."
        "Absolute path to a shared library with the"
        "kernels impl.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.6,
        help="Probability threshold for detections filtering"
        "(0.6 by default)",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        required=False,
        nargs="+",
        default=[],
        help="Argument to specify if any type of visulaization of bounding box,"
        "display of other stats are required. Multiple arguments (correspnding"
        "to diiferent model) can be chained. Possible arguments:"
        "fd: face detection bbox,"
        "fld: bboxs on both images,"
        "hpe: head pose (displays three angles),",
    )
    parser.add_argument(
        "-of",
        "--output_fname",
        required=True,
        type=str,
        default="outputs_fp32",
        help="name of the output files used for benchmarking,"
        "these are based on different precision values,"
        "it can take values: {outputs_fp32, outputs_fp16, outputs_int8}",
    )

    return parser


def infer_on_stream(args):
    # path to the input file
    input_file = args.input_path

    # visualization flags requested
    display_items = args.visualize

    # instantiate logger object
    logger = log.getLogger()

    # variables to store the input preprocessing times for various models
    fd_input_preprocessing_time = 0
    fld_input_preprocessing_time = 0
    hpe_input_preprocessing_time = 0
    ge_input_preprocessing_time = 0

    # open the input file
    if input_file == "CAM":
        inputfeeder = InputFeeder("cam")
    elif input_file.endswith(".jpg") or input_file.endswith("bmp"):
        inputfeeder = InputFeeder("image", input_file)
    elif input_file.endswith(".mp4"):
        inputfeeder = InputFeeder("video", input_file)
    else:
        assert os.path.isfile(input_file), "Input file doesn't exist...exiting!"
        sys.exit(1)

    # storing all the model paths in a dictionary
    model_paths = {
        "face_detection": args.face_detection,
        "facial_landmarks_detection": args.facial_landmarks,
        "head_pose_estimation": args.head_pose,
        "gaze_estimation": args.gaze,
    }
    # checking if all the model file paths are valid
    for model_name in model_paths.keys():
        if not os.path.isfile(model_paths[model_name] + ".xml"):
            logger.error(
                f"Path to the xml file for the model: {model_name} doesn't exist...exiting!"
            )
            sys.exit(1)

    # load data from input feeder
    inputfeeder.load_data()
    logger.error("Input feeder is loaded")
    # instantiating mouse controller
    mc = MouseController(precision="medium", speed="fast")

    ## instantiating and loading each model; storing the load time for each model
    # start time for face detection model
    logger.error("Loading the models...")
    fd_model_start = time.time()
    fd_model = FaceDetectionModel(
        model_paths["face_detection"],
        device=args.device,
        threshold=args.prob_threshold,
        extensions=args.cpu_extension,
    )
    fd_model.load_model()
    fd_model_load_time = time.time() - fd_model_start
    logger.error(
        f"Face detection model loaded in {(fd_model_load_time*1000):.3f} ms."
    )

    # start time for facial landmarks model
    fld_model_start = time.time()
    fld_model = FacialLandmarksModel(
        model_paths["facial_landmarks_detection"],
        device=args.device,
        extensions=args.cpu_extension,
    )
    fld_model.load_model()
    fld_model_load_time = time.time() - fld_model_start
    logger.error(
        f"Facial landmarks detection model loaded in {(fld_model_load_time*1000):.3f} ms."
    )

    # start time for head pose estimation model
    hpe_model_start = time.time()
    hpe_model = HeadPoseEstimationModel(
        model_paths["head_pose_estimation"],
        device=args.device,
        extensions=args.cpu_extension,
    )
    hpe_model.load_model()
    hpe_model_load_time = time.time() - hpe_model_start
    logger.error(
        f"Head pose estimation model loaded in {(hpe_model_load_time*1000):.3f} ms."
    )

    # start time for gaze estimation model
    ge_model_start = time.time()
    ge_model = GazeEstimationModel(
        model_paths["gaze_estimation"],
        device=args.device,
        extensions=args.cpu_extension,
    )
    ge_model.load_model()
    ge_model_load_time = time.time() - ge_model_start
    logger.error(
        f"Gaze estimation model loaded in {(ge_model_load_time*1000):.3f} ms."
    )
    all_models_load_time = time.time() - fd_model_start
    logger.error(
        f"Total load time of all the models is {(all_models_load_time*1000):.3f} ms."
    )
    logger.error("All models loaded successfully!")

    # load the models and check for unsupported layers
    logger.error("Checking for unsupported layers ...")
    for model_obj in (fd_model, fld_model, hpe_model, ge_model):
        #     model_obj.load_model()
        model_obj.check_model()
    logger.error("Done!")

    # keep track of the frames
    frame_number = 0

    # start inference time
    logger.error("Starting the inference on the input video...")
    start_inference_time = time.time()
    # iterate through the frames batch
    for flag, frame in inputfeeder.next_batch():
        if not flag:
            break
        # keep track of frames passed
        frame_number += 1
        key_pressed = cv2.waitKey(60)

        # detect the face in the frame
        face_coordinates, face_image, fd_preprocessing_time = fd_model.predict(
            frame.copy()
        )
        fd_input_preprocessing_time += fd_preprocessing_time
        if face_coordinates == 0:
            logger.error("No face is detected")
            continue

        # detect the head pose in the face image
        head_pose_angles, hpe_preprocessing_time = hpe_model.predict(face_image)
        hpe_input_preprocessing_time += hpe_preprocessing_time

        # get the left and right eye images
        (
            left_eye_image,
            right_eye_image,
            eye_coordinates,
            fld_preprocessing_time,
        ) = fld_model.predict(face_image)
        fld_input_preprocessing_time += fld_preprocessing_time

        # get the coordinates for mouse controller
        (*mouse_coordinates, ge_preprocessing_time) = ge_model.predict(
            left_eye_image, right_eye_image, head_pose_angles
        )
        ge_input_preprocessing_time += ge_preprocessing_time

        # check if display stats are requested, if so, show them
        if len(display_items) != 0:
            display_frame = frame.copy()
            if "fd" in display_items:
                cv2.rectangle(
                    display_frame,
                    (face_coordinates[0], face_coordinates[1]),
                    (face_coordinates[2], face_coordinates[3]),
                    (32, 32, 32),
                    3,
                )

            if "hpe" in display_items:
                # show yaw, pitch and roll angles on the frame
                text = f"yaw:{head_pose_angles[0]:.1f}, pitch:{head_pose_angles[1]:.1f}, roll:{head_pose_angles[2]:.1f}"
                cv2.putText(
                    display_frame,
                    text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5,
                    color=(255, 255, 255),
                    thickness=3,
                )

            if "fld" in display_items:
                # showing bbox on left eye
                display_face_frame = face_image.copy()
                cv2.rectangle(
                    display_face_frame,
                    (eye_coordinates[0][0], eye_coordinates[0][1]),
                    (eye_coordinates[0][2], eye_coordinates[0][3]),
                    (220, 20, 60),
                    2,
                )

                # showing bbox on right eye
                cv2.rectangle(
                    display_face_frame,
                    (eye_coordinates[1][0], eye_coordinates[1][1]),
                    (eye_coordinates[1][2], eye_coordinates[1][3]),
                    (220, 20, 60),
                    2,
                )

        if "fd" in display_items:
            # if face detection is present, show both original and face image frames side by side
            show_frame = np.hstack(
                (
                    cv2.resize(display_frame, (500, 500)),
                    cv2.resize(display_face_frame, (500, 500)),
                )
            )
        elif "fd" not in display_items:
            # if fd flag is not present just show the zoomed in version of face
            show_frame = cv2.resize(display_face_frame, (500, 500))
        else:
            # just show the original frame without bboxes
            show_frame = cv2.resize(frame, (500, 500))

        # show the ouput video frame
        cv2.imshow("video", show_frame)

        # if frame_number % 5 == 0:
        #     cv2.imshow("video", cv2.resize(display_frame, (500, 500)))

        # move the mouse pointer in the gaze direction of person
        mc.move(mouse_coordinates[0][0], mouse_coordinates[0][1])

        # break the stream if "Esc" key is pressed
        if key_pressed == 27:
            logger.error("Exit key is pressed...exiting!")
            break
    # calculating the total time taken to run the inference
    total_inference_time = round(time.time() - start_inference_time, 2)

    # calculating the frames per second;
    # multiplying by 10 to get the total frames, since stream is parsed in
    # the batches of 10
    frames_per_second = int(frame_number) * 10 / total_inference_time
    # calculate the average input preprocessing time of each model for whole stream
    fd_input_preprocessing_time /= frame_number
    fld_input_preprocessing_time /= frame_number
    hpe_input_preprocessing_time /= frame_number
    ge_input_preprocessing_time /= frame_number

    logger.error("Done performing inference!")
    logger.error(f"Total batches: {frame_number}")
    logger.error(f"Total inference time: {total_inference_time} seconds.")
    logger.error(f"fps: {frames_per_second:.2f} frames/second")
    logger.error(
        f"Average input preprocessing time for face detection: {(fd_input_preprocessing_time*1000):.3f} ms."
    )
    logger.error(
        f"Average input preprocessing time for facial landmarks detection: {(fld_input_preprocessing_time*1000):.3f} ms."
    )
    logger.error(
        f"Average input preprocessing time for head pose estimation: {(hpe_input_preprocessing_time*1000):.3f} ms."
    )
    logger.error(
        f"Average input preprocessing time for gaze estimation: {(ge_input_preprocessing_time*1000):.3f} ms."
    )

    # writing all the logs in a file, will be needed for benchmarking
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            (args.output_fname + ".txt"),
        ),
        "w",
    ) as f:
        f.write(
            f"Face detection model loaded in {(fd_model_load_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Facial landmarks detection model loaded in {(fld_model_load_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Head pose estimation model loaded in {(hpe_model_load_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Gaze estimation model loaded in {(ge_model_load_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Total load time of models is {(all_models_load_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Average input preprocessing time for face detection: {(fd_input_preprocessing_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Average input preprocessing time for facial detection: {(fld_input_preprocessing_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Average input preprocessing time for head pose estimation: {(hpe_input_preprocessing_time*1000):.3f} ms.\n"
        )
        f.write(
            f"Average input preprocessing time for gaze estimation: {(ge_input_preprocessing_time*1000):.3f} ms.\n"
        )
        f.write(f"Inference finished in {total_inference_time} seconds.\n")
        f.write(f"fps: {(frames_per_second):.2f} frames/second. \n")

    # closing the video stream and destroy all opencv windows
    inputfeeder.close()
    cv2.destroyAllWindows()


def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)


if __name__ == "__main__":
    main()
