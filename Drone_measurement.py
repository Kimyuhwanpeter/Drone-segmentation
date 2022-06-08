# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

class Measurement:
    def __init__(self, predict, label, shape, total_classes=23):
        self.predict = predict
        self.label = label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)
        label_indices = np.where(self.label != 0)
        self.label = np.squeeze(np.take(self.label, label_indices), 1)
        self.predict = np.squeeze(np.take(self.predict, label_indices), 1)

        predict_count = np.bincount(self.predict, minlength=self.total_classes)
        label_count = np.bincount(self.label, minlength=self.total_classes)
        label_count_indices = np.where(label_count != 0)

        temp = self.total_classes * np.array(self.label, dtype="int") + np.array(self.predict, dtype="int")  # Get category metrics
    
        temp_count = np.bincount(temp, minlength=self.total_classes*self.total_classes)
        cm = np.reshape(temp_count, [self.total_classes, self.total_classes])
        cm = np.diag(cm)
    
        U = label_count + predict_count - cm

        out = np.zeros((self.total_classes))
        miou = np.divide(cm, U, out=out, where=U != 0)
        each_iou = miou
        miou = np.nansum(miou) / len(label_count_indices)

        return miou, each_iou

#img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/label_images_semantic/000.png")
#img = tf.image.decode_png(img, 1)
#img = tf.image.resize(img, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#img = tf.image.convert_image_dtype(img, tf.uint8)

#m, e = Measurement(img, img, [512*512,], 23).MIOU()


