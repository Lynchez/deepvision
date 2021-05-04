import os
import cv2
from tensorflow.keras.utils import get_file

class GenderDetection():

    def __init__(self):

        proto_url = 'https://download.fastcv.net/config/gender_detection/gender_deploy.prototxt'
        model_url = 'https://github.com/arunponnusamy/fastcv-files/releases/download/v0.1/gender_net.caffemodel'
        save_dir = os.path.expanduser('~') + os.path.sep + '.fastcv' + os.path.sep + 'pre-trained'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.proto = get_file('gender_deploy.prototxt', proto_url,
                                cache_subdir=save_dir) 
        self.model = get_file('gender_net.caffemodel', model_url,
                              cache_subdir=save_dir)

        self.labels = ['male', 'female']
        self.mean = (78.4263377603, 87.7689143744, 114.895847746)

        print('[INFO] Initializing gender detection model ..')
        self.net = cv2.dnn.readNetFromCaffe(self.proto, self.model)


    def detect_gender(self, face):

        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), self.mean,
                                     swapRB=False)
        self.net.setInput(blob)
        preds = self.net.forward()

        return (self.labels, preds[0])        
