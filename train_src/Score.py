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
class Losser():
    def __init__(self):
        self.loss = []
    def add(self, loss_item):
        self.loss.append(loss_item)
    def mean(self):
        return sum(self.loss) / len(self.loss)
        