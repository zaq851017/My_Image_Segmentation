import numpy as np
from sklearn.metrics import fbeta_score
import os
import cv2
from tqdm import tqdm
class Scorer():
    def __init__(self, config):
        self.predict = []
        self.label = []
        self.t = config.threshold
    def add(self, predict, label):
        self.predict += predict.flatten().tolist()
        self.label += label.flatten().tolist()
    def f1(self):
        return fbeta_score(self.label, self.predict, beta=1)
    def f2(self):
        return fbeta_score(self.label, self.predict, beta=2)
    def iou(self):
        """
        temp_predict = np.array(self.predict)
        temp_GT = np.array(self.label)
        tp_fp = np.sum(temp_predict == 1)
        tp_fn = np.sum(temp_GT == 1)
        tp = np.sum((temp_predict == 1) * (temp_GT == 1))
        """
        tp_fp = self.predict.count(1)
        tp_fn = self.label.count(1)
        tp = [a*b for a,b in zip(self.predict, self.label)].count(1)
        if (tp_fp + tp_fn - tp) == 0:
            iou = 0
        else:
            iou = tp / (tp_fp + tp_fn - tp)
        return iou
class Losser():
    def __init__(self):
        self.loss = []
    def add(self, loss_item):
        self.loss.append(loss_item)
    def mean(self):
        return sum(self.loss) / len(self.loss)
        