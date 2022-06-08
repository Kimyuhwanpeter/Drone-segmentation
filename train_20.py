# -*- coding:utf-8 -*-
from model_14 import *
from random import random, shuffle
from tensorflow.keras import backend as K
from model_profiler import model_profiler
from Drone_measurement import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/train.txt",

                           "test_txt_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/test.txt",

                           "val_txt_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/val.txt",
                           
                           "tr_label_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/label_images_semantic/",
                           
                           "tr_image_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/original_images/",

                           "te_label_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/label_images_semantic/",
                           
                           "te_image_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/original_images/",

                           "val_label_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/label_images_semantic/",
                           
                           "val_image_path": "D:/[1]DB/[5]4th_paper_DB/Drone/archive/dataset/semantic_drone_dataset/original_images/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/398/398",
                           
                           "lr": 1e-4,

                           "min_lr": 1e-7,
                           
                           "epochs": 400,

                           "total_classes": 23,

                           "ignore_label": 0,

                           "batch_size": 2,

                           "sample_images": "/yuhwan/Edisk/yuhwan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/sample_images",

                           "save_checkpoint": "/yuhwan/Edisk/yuhwan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/checkpoint",

                           "save_print": "/yuhwan/Edisk/yuhwan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/train_out.txt",

                           "train_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_loss.txt",

                           "train_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_acc.txt",

                           "val_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_loss.txt",

                           "val_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_acc.txt",

                           "test_images": "C:/Users/Yuhwan/Downloads/test_images",

                           "train": True})
# void label is 0 --> 1클래스를 0으로 당겨서시작하자 (원래 0 classs는 무시)
optim = tf.keras.optimizers.Adam(FLAGS.lr)
labels_color_map = np.array([[0, 0, 0],[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],
                      [28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],
                      [70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],
                      [190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],
                      [9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],
                      [112, 150, 146],[2, 135, 115],[255, 0, 0]], np.uint8)

color_map = np.array([[128, 64, 128],[130, 76, 0],[0, 102, 0],[112, 103, 87],
                      [28, 42, 168],[48, 41, 30],[0, 50, 89],[107, 142, 35],
                      [70, 70, 70],[102, 102, 156],[254, 228, 12],[254, 148, 12],
                      [190, 153, 153],[153, 153, 153],[255, 22, 96],[102, 51, 0],
                      [9, 143, 150],[119, 11, 32],[51, 51, 0],[190, 250, 190],
                      [112, 150, 146],[2, 135, 115],[255, 0, 0]], np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.cast(img, tf.float32)
    img = tf.image.random_brightness(img, max_delta=50.) 
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    # img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3], seed=123)
    no_img = img
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
     #lab = tf.image.random_crop(lab, [FLAGS.img_size, FLAGS.img_size, 1], seed=123)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)
        
    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    #img = tf.clip_by_value(img, 0, 255)
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)
    alpha = np.reshape(alpha, [1, FLAGS.total_classes])

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
        # return (tf.keras.backend.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Weighted Sørensen Dice coefficient.
    
    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.reduce_sum(y_true * y_pred)
    false_neg = tf.reduce_sum(y_true * (1-y_pred))
    false_pos = tf.reduce_sum((1-y_true) * y_pred)
    
    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + const)
    
    return coef_val # alpha part need to be fix!!!!!

def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3, const=K.epsilon()):
    
    '''
    Focal Tversky Loss (FTL)
    
    focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    
    ----------
    Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved 
    attention u-net for lesion segmentation. In 2019 IEEE 16th International Symposium on Biomedical Imaging 
    (ISBI 2019) (pp. 683-687). IEEE.
    
    ----------
    Input
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases 
        gamma: tunable parameter within [1, 3].
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    
    # (Tversky loss)**(1/gamma) 
    loss_val = tf.math.pow((1-tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1/gamma)
    
    return loss_val

def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels, object_buf):

    with tf.GradientTape() as tape:
        
        logits = run_model(model, images, True)
        labels = tf.reshape(labels, [-1]).numpy()
        logits = tf.reshape(logits, [-1, FLAGS.total_classes])
        indices = tf.squeeze(tf.where(labels != 0), 1).numpy() # 0 is void label
        temp_labels = tf.gather(labels, indices) - 1
        temp_logits = tf.gather(logits, indices)

        distribution_loss = categorical_focal_loss(alpha=[object_buf])(tf.one_hot(temp_labels, FLAGS.total_classes), 
                                                                       tf.nn.softmax(temp_logits, -1))

        one_hot_labels = tf.one_hot(temp_labels, FLAGS.total_classes)
        object_indices = tf.squeeze(tf.where(object_buf != 0.), 1).numpy()
        temp_object_buf = tf.gather(object_buf, object_indices)
        temp_object_buf = tf.cast(temp_object_buf, tf.float32)

        cla_indices = [tf.squeeze(tf.where(one_hot_labels[:, i] == 1.), 1) for i in object_indices]

        temp_class = [tf.gather(one_hot_labels[:, idx], cla_indices[i]) for i, idx in enumerate(object_indices)]
        temp_logits = [tf.gather(temp_logits[:, idx], cla_indices[i]) for i, idx in enumerate(object_indices)]

        spatial_loss = [focal_tversky(temp_class[i], tf.nn.sigmoid(temp_logits[i]), alpha=temp_object_buf[i]) for i in range(len(temp_class))]
        spatial_loss = tf.reduce_mean(spatial_loss)

        total_loss = distribution_loss + spatial_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss

def main():

    model = modified_Unet_PP(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), nclasses=FLAGS.total_classes)
    prob = model_profiler(model, FLAGS.batch_size)
    model.summary()
    print(prob)

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        count = 0

        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)
        val_list = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.tr_image_path + data + ".jpg" for data in train_list]
        test_img_dataset = [FLAGS.te_image_path + data + ".jpg" for data in test_list]
        val_img_dataset = [FLAGS.val_image_path + data + ".jpg" for data in val_list]

        train_lab_dataset = [FLAGS.tr_label_path + data + ".png" for data in train_list]
        test_lab_dataset = [FLAGS.te_label_path + data + ".png" for data in test_list]
        val_lab_dataset = [FLAGS.val_label_path + data + ".png" for data in val_list]

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)
            tr_iter = iter(train_ge)

            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)

                class_imbal_labels_buf = 0.
                class_imbal_labels = batch_labels
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes + 1)
                    class_imbal_labels_buf += count_c_i_lab

                object_buf = class_imbal_labels_buf[1:]
                non_object_indices = np.where(object_buf != 0)[0]
                temp_object_buf = object_buf[non_object_indices]

                temp_object_buf = (np.max(temp_object_buf / np.sum(temp_object_buf)) + 1 - (temp_object_buf / np.sum(temp_object_buf)))
                temp_object_buf = tf.nn.softmax(temp_object_buf).numpy()
                if max(temp_object_buf) * 10. < 1:
                    temp_object_buf = temp_object_buf * 10.
                else:
                    temp_object_buf = temp_object_buf

                object_buf[non_object_indices] = temp_object_buf

                loss = cal_loss(model, batch_images, batch_labels, object_buf)

                if count % 10 == 0:
                    print("Epochs: {}, Loss = {} [{}/{}]".format(epoch, loss, step + 1, tr_idx))

                if count % 100 == 0:
                    logits = run_model(model, batch_images, False)
                    logits = tf.nn.softmax(logits, -1)
                    logits = tf.argmax(logits, -1)
                    logits = tf.cast(logits, tf.int32)
                    for i in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[i, :, :, 0], tf.int32).numpy()
                        output_ = logits[i, :, :]
                        final_output = output_

                        pred_mask_color = color_map[final_output]
                        label_mask_color = labels_color_map[label]

                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)

                count += 1

            tr_iter = iter(train_ge)
            miou = 0.
            each_iou = []
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False)
                    logits = tf.nn.softmax(logits, -1)
                    logits = tf.argmax(logits, -1)
                    logits = tf.cast(logits, tf.int32)

                    output_ = logits[0, :, :]
                    final_output = output_

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = tf.cast(batch_label, tf.int32)

                    miou_, each_iou_ = Measurement(predict=final_output,
                                       label=batch_label,
                                       shape=[FLAGS.img_size*FLAGS.img_size,],
                                       total_classes=FLAGS.total_classes).MIOU()
                    miou += miou_
                    each_iou += each_iou_

            final_miou = miou / len(train_img_dataset)
            final_each_iou = each_iou / len(train_img_dataset)
            print("train mIoU = %.4f (paved-area = %.4f, dirt = %.4f, grass = %.4f, gravel = %.4f, water = %.4f,rocks = %.4f, pool = %.4f, vegetation = %.4f, roof = %.4f, wall = %.4f, window = %.4f, door = %.4f, fence = %.4f, fence-pole = %.4f, person = %.4f, dog = %.4f, car = %.4f, bicyle = %.4f, tree = %.4f, bald-tree = %.4f, ar-marker = %.4f, obstacle = %.4f, conflicting = %.4f)".format(final_miou, final_each_iou[0], final_each_iou[1], final_each_iou[2], final_each_iou[3], final_each_iou[4], final_each_iou[5], final_each_iou[6], final_each_iou[7], \
                final_each_iou[8], final_each_iou[9], final_each_iou[10], final_each_iou[11], final_each_iou[12], final_each_iou[13], final_each_iou[14], \
                final_each_iou[15], final_each_iou[16], final_each_iou[17], final_each_iou[18], final_each_iou[19], final_each_iou[20], final_each_iou[21], final_each_iou[22]))

            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train mIoU: ")
            output_text.write("%.4f" % (final_miou))
            output_text.write(" (train each Iou: ")
            for i in range(FLAGS.total_classes):
                if i == FLAGS.total_classes - 1:
                    output_text.write("%.4f)" % (final_each_iou[FLAGS.total_classes - 1]))
                else:
                    output_text.write("%.4f, " % (final_each_iou[i]))
            output_text.write("\n")


            val_ge = tf.data.Dataset.from_tensor_slices((val_img_dataset, val_lab_dataset))
            val_ge = val_ge.map(test_func)
            val_ge = val_ge.batch(1)
            val_ge = val_ge.prefetch(tf.data.experimental.AUTOTUNE)

            val_iter = iter(val_ge)
            miou = 0.
            each_iou = []
            for i in range(len(val_img_dataset)):
                batch_images, batch_labels = next(val_iter)
                logits = run_model(model, batch_images, False)
                logits = tf.nn.softmax(logits, -1)
                logits = tf.argmax(logits, -1)
                logits = tf.cast(logits, tf.int32)

                output_ = logits[0, :, :]
                final_output = output_

                batch_label = tf.cast(batch_labels[0, :, :, 0], tf.uint8).numpy()
                batch_label = tf.cast(batch_label, tf.int32)

                miou_, each_iou_ = Measurement(predict=final_output,
                                    label=batch_label,
                                    shape=[FLAGS.img_size*FLAGS.img_size,],
                                    total_classes=FLAGS.total_classes).MIOU()
                miou += miou_
                each_iou += each_iou_

            final_miou = miou / len(val_img_dataset)
            final_each_iou = each_iou / len(val_img_dataset)
            print("val mIoU = %.4f (paved-area = %.4f, dirt = %.4f, grass = %.4f, gravel = %.4f, water = %.4f,rocks = %.4f, pool = %.4f, vegetation = %.4f, roof = %.4f, wall = %.4f, window = %.4f, door = %.4f, fence = %.4f, fence-pole = %.4f, person = %.4f, dog = %.4f, car = %.4f, bicyle = %.4f, tree = %.4f, bald-tree = %.4f, ar-marker = %.4f, obstacle = %.4f, conflicting = %.4f)".format(final_miou, final_each_iou[0], final_each_iou[1], final_each_iou[2], final_each_iou[3], final_each_iou[4], final_each_iou[5], final_each_iou[6], final_each_iou[7], \
                final_each_iou[8], final_each_iou[9], final_each_iou[10], final_each_iou[11], final_each_iou[12], final_each_iou[13], final_each_iou[14], \
                final_each_iou[15], final_each_iou[16], final_each_iou[17], final_each_iou[18], final_each_iou[19], final_each_iou[20], final_each_iou[21], final_each_iou[22]))

            output_text.write("val mIoU: ")
            output_text.write("%.4f" % (final_miou))
            output_text.write(" (val each Iou: ")
            for i in range(FLAGS.total_classes):
                if i == FLAGS.total_classes - 1:
                    output_text.write("%.4f)" % (final_each_iou[FLAGS.total_classes - 1]))
                else:
                    output_text.write("%.4f, " % (final_each_iou[i]))
            output_text.write("\n")


            test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
            test_ge = test_ge.map(test_func)
            test_ge = test_ge.batch(1)
            test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

            test_iter = iter(test_ge)
            miou = 0.
            each_iou = []
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                logits = run_model(model, batch_images, False)
                logits = tf.nn.softmax(logits, -1)
                logits = tf.argmax(logits, -1)
                logits = tf.cast(logits, tf.int32)

                output_ = logits[0, :, :]
                final_output = output_

                batch_label = tf.cast(batch_labels[0, :, :, 0], tf.uint8).numpy()
                batch_label = tf.cast(batch_label, tf.int32)

                miou_, each_iou_ = Measurement(predict=final_output,
                                    label=batch_label,
                                    shape=[FLAGS.img_size*FLAGS.img_size,],
                                    total_classes=FLAGS.total_classes).MIOU()
                miou += miou_
                each_iou += each_iou_

            final_miou = miou / len(test_img_dataset)
            final_each_iou = each_iou / len(test_img_dataset)
            print("test mIoU = %.4f (paved-area = %.4f, dirt = %.4f, grass = %.4f, gravel = %.4f, water = %.4f,rocks = %.4f, pool = %.4f, vegetation = %.4f, roof = %.4f, wall = %.4f, window = %.4f, door = %.4f, fence = %.4f, fence-pole = %.4f, person = %.4f, dog = %.4f, car = %.4f, bicyle = %.4f, tree = %.4f, bald-tree = %.4f, ar-marker = %.4f, obstacle = %.4f, conflicting = %.4f)".format(final_miou, final_each_iou[0], final_each_iou[1], final_each_iou[2], final_each_iou[3], final_each_iou[4], final_each_iou[5], final_each_iou[6], final_each_iou[7], \
                final_each_iou[8], final_each_iou[9], final_each_iou[10], final_each_iou[11], final_each_iou[12], final_each_iou[13], final_each_iou[14], \
                final_each_iou[15], final_each_iou[16], final_each_iou[17], final_each_iou[18], final_each_iou[19], final_each_iou[20], final_each_iou[21], final_each_iou[22]))

            output_text.write("test mIoU: ")
            output_text.write("%.4f" % (final_miou))
            output_text.write(" (test each Iou: ")
            for i in range(FLAGS.total_classes):
                if i == FLAGS.total_classes - 1:
                    output_text.write("%.4f)" % (final_each_iou[FLAGS.total_classes - 1]))
                else:
                    output_text.write("%.4f, " % (final_each_iou[i]))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/drone_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)

if __name__ == "__main__":
    main()
