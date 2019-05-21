# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img):
        image = cv2.resize(img, (self.input_shape[0], self.input_shape[1]), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feat = self.model.predict(image)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat
