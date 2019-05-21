
from yolo import YOLO
from extract_cnn_vgg16_keras import VGGNet
import h5py

import os
import cv2
import time
import numpy as np

Model_Init_Flag = 0

class LOGO():
    def __init__(self):

        self.yolo = YOLO()
        self.model = VGGNet()
        self.feats, self.imgNames = self.read_databae()

    def read_databae(self):
        h5f = h5py.File('logo_feature.h5', 'r')
        # feats = h5f['dataset_1'][:]
        feats = h5f['dataset_1'][:]
        imgNames = h5f['dataset_2'][:]
        h5f.close()
        return feats, imgNames

    def get_imlist(self,path):
        return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

    def create_database(self,data_path):
        db = data_path
        img_list = self.get_imlist(db)
        print("--------------------------------------------------")
        print("         feature extraction starts")
        print("--------------------------------------------------")
        feats = []
        names = []
        for i, img_path in enumerate(img_list):
            img = cv2.imread(img_path)
            norm_feat = self.model.extract_feat(img)
            img_name = os.path.split(img_path)[1]
            feats.append(norm_feat)
            names.append(img_name)
            print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
        feats = np.array(feats)

        # directory for storing extracted features
        output = 'logo_feature.h5'
        print("--------------------------------------------------")
        print("      writing feature extraction results ...")
        print("--------------------------------------------------")
        h5f = h5py.File(output, 'w')
        h5f.create_dataset('dataset_1', data = feats)

        h5f.create_dataset('dataset_2', data = np.string_(names))
        h5f.close()
        return True


    def search_img(self,query):
        # read in indexed images' feature vectors and corresponding image names
        print("--------------------------------------------------")
        print("               searching starts")
        print("--------------------------------------------------")

        # extract query image's feature, compute simlarity score and sort
        t2 = time.time()
        img = cv2.imread(query)
        image, logos = self.yolo.detect_image(img)

        for logo in logos:
            t4 = time.time()
            queryVec = self.model.extract_feat(logo)
            scores = np.dot(queryVec, self.feats.T)
            rank_ID = np.argsort(scores)[::-1]
            rank_score = scores[rank_ID]
            t5 = time.time() - t4
            print('t5=' + str(t5))
            print("-------------------------------------------")
            print(rank_ID)
            print(rank_score)
            print("-------------------------------------------")

            # number of top retrieved images to show
            imlist = [self.imgNames[index] for index in rank_ID[0:1]]
            file ='database/'+str(imlist)[3:-2]
            t3 = time.time() - t2
            print('t3=' + str(t3))
            print(rank_score[0])
            return file, t3, rank_score[0]
