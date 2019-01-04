import os
import ast
import cv2
import json
import random

import numpy as np
import pandas as pd
import keras
from keras.metrics import top_k_categorical_accuracy
from keras.utils import Sequence

DP_DIR = '../input/shuffled_csv'
INPUT_DIR = '../input'

NCSVS = 100
NCATS = 340

def check_dirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("%s didn't exist, created successfully!")


def print_save_HParams(hps, save_path):
    hps_dict = vars(hps)
    for key, val in hps_dict.items():
        print('%s = %s' % (key, str(val)))
    if not hps.debug:
        with open(save_path, 'w') as outfile:
            outfile.write(json.dumps(hps_dict, indent=4))
        print("HParams %s json file saved sucessfully!" % hps.tag)

def f2cat(filename: str) -> str:  # 强行定义参数类型和返回类型 
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_csv'))
    return sorted([f2cat(f) for f in files], key=str.lower)


def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k] 

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

 
    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def draw_cv2(raw_strokes, size=256, lw=6, time_color=True, base_size=256, point_drop_prob=0.0, channel=1):
    """
    将一个涂鸦数据转换成一张黑百图, (size, size, channel)
    time_color: 若为true, 则每一笔颜色不一样,随着时间推移而变浅. 若为false,则颜色都为纯黑
    lw: 线段宽度
    point_drop_prob: 以此概率丢弃掉point, 0.05 - 0.15差不多
    channel: 必须为1或3

    """
    if channel == 1:
        img = np.zeros((base_size, base_size), np.uint8)
    
    elif channel == 3:
        img = np.zeros((base_size, base_size, 3), np.uint8)
    else:
        assert False
    
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            if point_drop_prob > 0.0 and random.randint(0,100) < point_drop_prob * 100:
                continue
            color = 255 - min(t, 10) * 13 if time_color else 255
            if channel == 3:
                color = (color, color, color)
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw) # color是一个标量所以生成黑白图
    if size != base_size:
        return cv2.resize(img, (size, size))    
    else:
        return img


def df_to_gray_image_array(df, size, preprocess_func=None, lw=6, time_color=True, base_size=256, point_drop_prob=0.0, 
                           channel=1):
    """
    将原始df里的所有sketch数据转换成(size x size x channel)的黑白图片
    preprocess_func: 对图像数据进行处理的函数
    """
    df['drawing_list'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, channel))
    for i, raw_strokes in enumerate(df.drawing_list.values):
        if channel == 3:
            x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, base_size=base_size,
                                     point_drop_prob = point_drop_prob, channel=channel)
        else:
            x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, base_size=base_size,
                                     point_drop_prob = point_drop_prob, channel=channel)
    del df['drawing_list']
    if preprocess_func is not None:
        x = preprocess_func(x).astype(np.float32)
    return x


def image_generator_xd(size, batchsize, ks, preprocess_func=None, lw=6, time_color=True, base_size=256, 
                       point_drop_prob=0.0, channel=1):
    """
    train,valid数据生成器
    """
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_%d_%d.csv.gz'%(k, NCSVS))
            for df in pd.read_csv(filename, chunksize=batchsize): # 利用参数chunksize可以分块读取大文件
                x = df_to_gray_image_array(df, size=size, preprocess_func=preprocess_func, 
                                           lw=lw, time_color=time_color, base_size=base_size, 
                                           point_drop_prob=point_drop_prob, channel=channel)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y


def test_image_generator(test_df_path, size, batchsize, preprocess_func=None, lw=6, time_color=True, base_size=256, 
                         point_drop_prob=0.0, channel=1):
    """
    test data generator
    """
    for df in pd.read_csv(test_df_path, chunksize=batchsize): # 利用参数chunksize可以分块读取大文件
        x = df_to_gray_image_array(df=df, size=size, preprocess_func=preprocess_func,
                                   lw=lw, time_color=time_color, base_size=base_size, 
                                   point_drop_prob=point_drop_prob, channel=channel)
        yield x

        
class Iamge_Sequence(Sequence):

    def __init__(self, size, batchsize, ks, batches_per_file, preprocess_func=None, lw=6, time_color=True, base_size=256, 
                       point_drop_prob=0.0, channel=1):
        self.size = size
        self.batchsize = batchsize
        self.batches_per_file = batches_per_file
        self.files = [os.path.join(DP_DIR, 'train_%d_%d.csv.gz'%(k, NCSVS)) for k in ks]
        self.preprocess_func = preprocess_func
        self.lw = lw
        self.time_color = time_color
        self.base_size = base_size
        self.point_drop_prob = point_drop_prob
        self.channel = channel
        

    def __len__(self):
        return int(self.batches_per_file * len(self.files))

    def __getitem__(self, idx):
        file = self.files[idx // self.batches_per_file]
        start_row = (idx % self.batches_per_file) * self.batchsize
        
        # 读取从start_row行开始的batchsize行数据(计数从0开始的)
        batch_rows = pd.read_csv(file, skiprows = start_row, nrows=self.batchsize) # skiprows = 0的话就相当于正常读取，不跳过
        tmp = pd.read_csv(file, nrows=1) # tmp的唯一作用就是它的columns
        batch_rows.columns = tmp.columns
        batch_x = df_to_gray_image_array(batch_rows, size=self.size, preprocess_func=self.preprocess_func, 
                                           lw=self.lw, time_color=self.time_color, base_size=self.base_size, 
                                           point_drop_prob=self.point_drop_prob, channel=self.channel)
        batch_y = keras.utils.to_categorical(batch_rows.y, num_classes=NCATS)
        

        return batch_x, batch_y

