# -*- coding: utf-8 -*-
import os

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot

import pandas as pd
import tensorflow as tf
import random
import math
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn import metrics

import h5py

def read_data(path, padding="pad_sequence", maxlen=1500):
    pssm_files = os.listdir(path)
    data = []
    labels = []
    for pssm_file in pssm_files:
        df = pd.read_csv(path + pssm_file, sep=',', header=None)
        df = np.asarray(df, dtype=np.float32)
        # df = normalize(df.T)
        df = df.T
        if padding == "pad_sequence":
            df = sequence.pad_sequences(df, maxlen=maxlen, padding='post', truncating='post', dtype='float32', value=0.0)
        elif padding == "same":
            df = pad_same(df, maxlen=maxlen)
        data.append(df)
        labels.append(pssm_file.split('.')[0].split('_')[1])
    # data = np.asarray(data, dtype=np.float32)
    return data, labels


def get_model_name(k):
    return 'model_' + str(k) + '.h5'

def balance_data(data, labels, random_state):
    posi = []
    nega = []
    balanced_data = []
    balanced_labels = []

    for i in range(len(data)):
        if labels[i] == 1:
            posi.append(data[i])
        else:
            nega.append(data[i])

    random.Random(random_state).shuffle(posi)    

    if len(posi) < len(nega):
        tmp = int(len(nega) / len(posi))
        j = 0
        for i in range(len(nega)):
            if i % tmp == 0:
                if j < len(posi):
                    balanced_data.append(posi[j])
                    balanced_labels.append(1)
                    j += 1
            balanced_data.append(nega[i])
            balanced_labels.append(0)
    else:
        tmp = int(len(posi) / len(nega))
        j = 0
        for i in range(len(posi)):
            if i % tmp == 0:
                if j < len(nega):
                    balanced_data.append(nega[j])
                    balanced_labels.append(0)
                    j += 1
            balanced_data.append(posi[i])
            balanced_labels.append(1)

    return np.asarray(balanced_data), np.asarray(balanced_labels)

def median(models, data):
    pre = []
    for model in models:
        pre.append(model.predict(data))
        print(model.predict(data).shape)
        #pre.append(model.predict_proba(data))

    med = []
    for i in range(len(pre[0])):
        b = []
        for j in range(len(pre)):
            b.append(pre[j][i])
        med.append(np.median(b, axis=0))
    return np.asarray(med)
    
def plot_loss(history, i):
    f = plt.figure()
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss', fontsize=24)
    plt.xlabel('epochs', fontsize=24)
    plt.legend(['train loss', 'val loss'], loc='upper right', prop={'size': 20})
    plt.show()

def plot_accuracy(history, i):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.savefig('saved_plots/accuracy_{}.png'.format(i))
    # plt.clf()

def plot_sensitivity(history, i):
    plt.plot(history['sensitivity'])
    plt.plot(history['val_sensitivity'])
    plt.title('model sensitivity')
    plt.ylabel('sensitivity')
    plt.xlabel('epochs')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()
    # plt.savefig('saved_plots/sensitivity_{}.png'.format(i))
    # plt.clf()


def plot_specificity(history, i):
    plt.plot(history['specificity'])
    plt.plot(history['val_specificity'])
    plt.title('model specificity')
    plt.ylabel('specificity')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.savefig('saved_plots/specificity_{}.png'.format(i))
    # plt.clf()


def save_h5py(data, labels, path, name):
    f = h5py.File(path + name + '.h5', 'w')
    f.create_dataset(name, data=data)
    f.create_dataset('labels', data=labels)
    f.close()


def read_h5py(path, name):
    file = h5py.File(path + name + '.h5', 'r')
    data = np.asarray(file[name], dtype=np.float32)
    labels = np.asarray(file['labels'], dtype=np.float32)
    return data, labels


def predict(model, data_loader, cuda=True):
    if cuda:
        model.cuda()

    y_true = []
    y_pred = []
    for idx, (e, target) in enumerate(data_loader):
        if cuda:
            e, target = e.cuda(), target.cuda()        
        pred = model(e)
        y_true.extend(target.data.cpu().numpy())
        y_pred.extend(pred.data.cpu().numpy())
    return np.asarray(y_true), np.asarray(y_pred)


def get_confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])


def sensitivity(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    y_pred_bin = tf.cast(y_pred_bin, tf.float32)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)

    sensitivity = TP / (TP + FN + K.epsilon())
    return sensitivity


def specificity(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    y_pred_bin = tf.cast(y_pred_bin, tf.float32)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)

    specificity = TN / (TN + FP + K.epsilon())
    return specificity


def mcc(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    y_pred_bin = tf.cast(y_pred_bin, tf.float32)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)

    MCC = (TP * TN) - (FP * FN)
    MCC /= (tf.math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + K.epsilon())
    return MCC


def acc(y_true, y_pred):
    y_pred_bin = K.argmax(y_pred, axis=-1)
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_bin, num_classes=2)
    # as Keras Tensors
    TP = tf.cast(confusion_matrix[1, 1], dtype=tf.float32)
    FN = tf.cast(confusion_matrix[1, 0], dtype=tf.float32)
    TN = tf.cast(confusion_matrix[0, 0], dtype=tf.float32)
    FP = tf.cast(confusion_matrix[0, 1], dtype=tf.float32)

    acc = (TP + TN) / (TP + TN + FP + FN + K.epsilon())
    return acc


def auc(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    m = tf.keras.metrics.AUC()
    m.update_state(y_true, y_pred[:, 1])
    return m.result().numpy()
