# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.utils.bottleneck as bottleneck
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.profiler import profile, record_function, ProfilerActivity

import sys
import time
sys.path.append('/root/DeepCTR-Torch')

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from get_criteo import get_criteo_dataset

if __name__ == "__main__":
    # data = pd.read_csv('./criteo_sample.txt')
    
    # parse arguments from sys
    # sys[1] = 'part' or 'total' or 'sample'
    # sys[2] = 'muti' or 'single'
    
    if len(sys.argv) != 3:
        print('Usage: python run_classification_criteo.py [data_type] [device_type]')
        sys.exit(1)
    data_type = sys.argv[1]
    device_type = sys.argv[2]
    assert data_type in ['part', 'total', 'sample']
    assert device_type in ['multi', 'single']
    
    start_time = time.time()
    data = get_criteo_dataset(data_type)
    if device_type == 'multi':
        gpus = [0, 1, 2, 3]
    else:
        gpus = None
    
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir='runs/xDeepFM-{}-{}'.format(data_type, device_type))
    
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary',
    #                l2_reg_embedding=1e-5, device=device)

    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
                task='binary', l2_reg_embedding=1e-5, device=device, gpus=gpus)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    load_time = time.time()
    print('load time: ', load_time - start_time)
    history = model.fit(train_model_input, train[target].values, batch_size=4096, epochs=10, verbose=2,
                        validation_split=0.2, writer=writer)
    pred_ans = model.predict(test_model_input, 256)
    # torch.save(model.state_dict(), 'model.pth')
    eval_time = time.time()
    print("")
    print("CIN component compute time: ", model.cin_compute_time)
    print("DNN component compute time: ", model.dnn_compute_time)
    print('train time: ', eval_time - load_time)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
