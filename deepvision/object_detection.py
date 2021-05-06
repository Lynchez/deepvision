import cv2
import os
import numpy as np
from .utils import download_file

initialize = True
net = None
dest_dir = os.path.expanduser('~') + os.path.sep + '.deepvision' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'
classes = None
COLORS = np.random.uniform(0, 255, size=(80, 3))

def populate_class_labels():

    class_file_name = 'yolov3_classes.txt'
    class_file_abs_path = dest_dir + os.path.sep + class_file_name
    url = 'https://github.com/Lynchez/deepvision/raw/master/model_data/coco_classes.txt'
    if not os.path.exists(class_file_abs_path):
        download_file(url=url, file_name=class_file_name, dest_dir=dest_dir)
    f = open(class_file_abs_path, 'r')
    classes = [line.strip() for line in f.readlines()]

    return classes

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bbox(img, bbox, labels, confidence, colors=None, write_conf=False):
    """A method to apply a box to the image
    Args:
        img: An image in the form of a numPy array
        bbox: An array of bounding boxes
        labels: An array of labels
        colors: An array of colours the length of the number of targets(80)
        write_conf: An option to write the confidences to the image
    """

    global COLORS
    global classes

    if classes is None:
        classes = populate_class_labels()
    
    for i, label in enumerate(labels):

        if colors is None:
            color = COLORS[classes.index(label)]            
        else:
            color = colors[classes.index(label)]

        if write_conf:
            label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'

        cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)

        cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img
    
def detect_common_objects(image, confidence=0.5, nms_thresh=0.3, model='yolov4-tiny', enable_gpu=False):
    """A method to detect common objects
    Args:
        image: A colour image in a numpy array
        confidence: A value to filter out objects recognised to a lower confidence score
        nms_thresh: An NMS value
        model: The detection model to be used, supported models are: yolov3, yolov3-tiny, yolov4, yolov4-tiny
        enable_gpu: A boolean to set whether the GPU will be used

    """

    Height, Width = image.shape[:2]
    scale = 0.00392

    global classes
    global dest_dir

    if model == 'yolov3-tiny':
        config_file_name = 'yolov3-tiny.cfg'
        cfg_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg"
        weights_file_name = 'yolov3-tiny.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    elif model == 'yolov4':
        config_file_name = 'yolov4.cfg'
        cfg_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
        weights_file_name = 'yolov4.weights'
        weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    elif model == 'yolov4-tiny':
        config_file_name = 'yolov4-tiny.cfg'
        cfg_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
        weights_file_name = 'yolov4-tiny.weights'
        weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)      

    else:
        config_file_name = 'yolov3.cfg'
        cfg_url = 'https://github.com/Lynchez/deepvision/raw/master/model_data/yolov3.cfg'
        weights_file_name = 'yolov3.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)    

    config_file_abs_path = dest_dir + os.path.sep + config_file_name
    weights_file_abs_path = dest_dir + os.path.sep + weights_file_name    
    
    if not os.path.exists(config_file_abs_path):
        download_file(url=cfg_url, file_name=config_file_name, dest_dir=dest_dir)

    if not os.path.exists(weights_file_abs_path):
        download_file(url=weights_url, file_name=weights_file_name, dest_dir=dest_dir)    

    global initialize
    global net

    if initialize:
        classes = populate_class_labels()
        net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)
        initialize = False

    # enables opencv dnn module to use CUDA on Nvidia card instead of cpu
    if enable_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_conf = scores[class_id]
            if max_conf > confidence:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - (w / 2)
                y = center_y - (h / 2)
                class_ids.append(class_id)
                confidences.append(float(max_conf))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

    bbox = []
    label = []
    conf = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        bbox.append([int(x), int(y), int(x+w), int(y+h)])
        label.append(str(classes[class_ids[i]]))
        conf.append(confidences[i])
        
    return bbox, label, conf

class YOLO:

    def __init__(self, weights, config, labels, version='yolov3'):

        print('[INFO] Initializing YOLO ..')

        self.config = config
        self.weights = weights
        self.version = version

        with open(labels, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.colors = np.random.uniform(0, 255, size=(len(self.labels), 3))

        self.net = cv2.dnn.readNet(self.weights, self.config)
    
        layer_names = self.net.getLayerNames()
    
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def detect_objects(self, image, confidence=0.5, nms_thresh=0.3,
                       enable_gpu=False):

        if enable_gpu:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        Height, Width = image.shape[:2]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True,
                                     crop=False)    

        self.net.setInput(blob)

        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                max_conf = scores[class_id]
                if max_conf > confidence:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - (w / 2)
                    y = center_y - (h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(max_conf))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

        bbox = []
        label = []
        conf = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            bbox.append([int(x), int(y), int(x+w), int(y+h)])
            label.append(str(self.labels[class_ids[i]]))
            conf.append(confidences[i])
            
        return bbox, label, conf


    def draw_bbox(self, img, bbox, labels, confidence, colors=None, write_conf=False):

        if colors is None:
            colors = self.colors

        for i, label in enumerate(labels):

            color = colors[self.labels.index(label)]

            if write_conf:
                label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'

            cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)

            cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class TFlite:
    def __init__(self, tflite_model="default", coconames="default"):
        import tensorflow as tf
        config_file_name = 'coconames.txt'
        cfg_url = 'https://github.com/Lynchez/deepvision/raw/master/model_data/coconames.txt'
        weights_file_name = 'detect.tflite'
        weights_url = 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite3/detection/default/1?lite-format=tflite'
        dest_dir = os.path.expanduser('~') + os.path.sep + '.deepvision' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'

        config_file_abs_path = dest_dir + os.path.sep + config_file_name
        weights_file_abs_path = dest_dir + os.path.sep + weights_file_name   

        if not os.path.exists(weights_file_abs_path):
            download_file(url=cfg_url, file_name=config_file_name, dest_dir=dest_dir)
            download_file(url=weights_url, file_name=weights_file_name, dest_dir=dest_dir)

        if tflite_model == "default":
            self.interpreter = tf.lite.Interpreter(model_path=weights_file_abs_path)
            with open(config_file_abs_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_model)
            with open(coconames, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # Test the TensorFlow Lite model on random input data.
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5

    def detect_objects(self, img, min_conf_threshold = 0.5):
        frame_resized = cv2.resize(img, (self.width, self.height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Bounding box coordinates of detected objects
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * self.height)))
                xmin = int(max(1, (boxes[i][1] * self.width)))
                ymax = int(min(self.height, (boxes[i][2] * self.height)))
                xmax = int(min(self.width, (boxes[i][3] * self.width)))
                cv2.rectangle(frame_resized, (xmin, ymin),(xmax, ymax), (10, 255, 0), 2)
                object_name = self.labels[int(classes[i])]
                cv2.putText(frame_resized, object_name, (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,35,95), 2)

        return frame_resized
