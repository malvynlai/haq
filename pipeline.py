import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
import pickle


def available_ingredients(filepath):
    root_dir = Path.cwd()
    onnxruntime_dir = Path(ort.__file__).parent

    hexagon_driver = Path.joinpath(onnxruntime_dir,'capi','QnnHtp.dll')
    qnn_provider_options = {
        'backend_path':hexagon_driver,
    }

    image = Image.open(filepath)


    with open('imagenet_labels.pkl', 'rb') as file:
        imagenet_labels = pickle.load(file)


    with open('coco_labels.pkl', 'rb') as file:
        yolo_labels = pickle.load(file)

    model_name = 'yolox-yolo-x-float.onnx'
    # model_name = 'detr_resnet101.onnx'
    yolo_model = Path.joinpath(root_dir,'models',model_name)

    model_name = 'googlenet-googlenet-float.onnx'
    model_name = 'inception_v3.onnx'
    googlenet_model = Path.joinpath(root_dir,'models',model_name)

    yolo_session = ort.InferenceSession(yolo_model,
                                providers=[('QNNExecutionProvider',qnn_provider_options),'CPUExecutionProvider']
                                )

    googlenet_session = ort.InferenceSession(googlenet_model,
                                providers=[('QNNExecutionProvider',qnn_provider_options),'CPUExecutionProvider']
                                )


    yolo_inputs = yolo_session.get_inputs()
    yolo_outputs = yolo_session.get_outputs()
    yolo_input_0 = yolo_inputs[0]
    yolo_output_0 = yolo_outputs[0]


    googlenet_inputs = googlenet_session.get_inputs()
    googlenet_outputs = googlenet_session.get_outputs()
    googlenet_input_0 = googlenet_inputs[0]
    googlenet_output_0 = googlenet_outputs[0]



    yolo_expected_shape = yolo_input_0.shape
    googlenet_expected_shape = googlenet_input_0.shape

    def transform_numpy_opencv(image: np.ndarray, 
                            expected_shape: Tuple[int, int],
                            ) -> Tuple[np.ndarray, np.ndarray]:

        height, width = expected_shape[2], expected_shape[3]
        resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)

        resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)

        float_image = resized_image.astype(np.float32) / 255.0
        chw_image = np.transpose(float_image, (2,0,1)) # HWC -> CHW

        return (float_image,chw_image)



    ###########################################################################
    ## This is for scaling purposes ###########################################
    ###########################################################################
    input_image_height, input_image_width = yolo_expected_shape[2], yolo_expected_shape[3]

    hwc_frame_processed, chw_frame = transform_numpy_opencv(np.array(image), yolo_expected_shape)

    ########################################################################
    ## INFERENCE ###########################################################  
    ########################################################################
    inference_frame = np.expand_dims(chw_frame, axis=0)
    outputs = yolo_session.run(None, {yolo_input_0.name:inference_frame})

    item_label_list = []

    yolo_tags = outputs[2][0,(np.where(outputs[1].squeeze()>.15)[0])].astype(np.int32)
    for tag in yolo_tags:
        item_label_list.append(yolo_labels[tag])

    object_list = outputs[0][0,(np.where(outputs[1].squeeze()>0.15)[0])].astype(np.int32)
    object_list = np.unique(object_list,axis=0)
    object_list = np.clip(object_list,min=0,max=None).astype(np.int32)


    object_image_list = []
    for object in object_list:
        x, y, w, h = object
        cropped_image = hwc_frame_processed[y:h, x:w]*255
        try:
            _, chw_frame = transform_numpy_opencv(np.array(cropped_image), googlenet_expected_shape)
            object_image_list.append(chw_frame)
        except:
            pass
    object_image_list = np.array(object_image_list)


    for object_image in object_image_list:

        inference_frame = np.expand_dims(object_image, axis=0)

        outputs = googlenet_session.run(None, {googlenet_input_0.name:inference_frame})
        class_idx = outputs[0].argmax()
        item_label_list += imagenet_labels[class_idx].split(',')

    item_label_list = list(set(item_label_list))
    return item_label_list

if __name__ == '__main__':
    print(available_ingredients('darrien-staton-unsplash.jpg'))