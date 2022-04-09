# -*- coding: utf-8 -*-
import json

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from helpers import *

if __name__ == '__main__':
    non_TF_pssm = 'non_TF_independent'
    TF_pssm = 'TF_independent'
    path = './pssm/'
    # read data
    data_non_TF_pssm, _ = read_data(path + non_TF_pssm + '/', padding="pad_sequence", maxlen=1500)
    labels_non_TF_pssm = np.zeros(len(data_non_TF_pssm))
    data_TF_pssm, _ = read_data(path + TF_pssm + '/', padding="pad_sequence", maxlen=1500)
    labels_TF_pssm = np.ones(len(data_TF_pssm))
    
    data = np.append(data_non_TF_pssm, data_TF_pssm, axis=0)
    labels = np.append(labels_non_TF_pssm, labels_TF_pssm, axis=0)

    print(data.shape)
    data = np.expand_dims(data, axis=-1).astype(np.float32)
    path = "./saved_models/"
    model_paths = os.listdir(path)
    model = []
    for model_path in model_paths:
        model.append(keras.models.load_model(path + model_path,
            custom_objects={"sensitivity": sensitivity,
            "specificity": specificity, "mcc": mcc }, compile=False))

    i = 0
    a = []
    b = []
    for i in range(len(model)):
        pre = model[i].predict(data)
        print("model: " + str(i))
        sen = sensitivity(labels, pre)
        spe = specificity(labels, pre)
        accc = acc(labels, pre)
        mccc = mcc(labels, pre)
        aucc = auc(labels, pre)
        b.append(math.floor(sen * 100000) / 100000)
        b.append(math.floor(spe * 100000) / 100000)
        b.append(math.floor(accc * 100000) / 100000)
        b.append(math.floor(mccc * 100000) / 100000)
        b.append(math.floor(aucc * 100000) / 100000)
        a.append(b)
        b = []
        i += 1

    med = median(model, data)
    sen = sensitivity(labels, med)
    spe = specificity(labels, med)
    accc = acc(labels, med)
    mccc = mcc(labels, med)
    aucc = auc(labels, med)
    b.append(math.floor(sen * 100000) / 100000)
    b.append(math.floor(spe * 100000) / 100000)
    b.append(math.floor(accc * 100000) / 100000)
    b.append(math.floor(mccc * 100000) / 100000)
    b.append(math.floor(aucc * 100000) / 100000)
    a.append(b)
    b = []    
    pd_result = pd.DataFrame(a)
    pd_result.columns =["Sensitivity", "Specificity", "Accuracy", "MCC", "AUC"]
    pd_result.index.names = ['Model']
    pd_result.to_csv('./test.csv')
