import os
import ast
import cv2
import random
import pandas as pd
import numpy as np
from PIL import Image
import datetime as dt

import torch
from torch.utils.data import Dataset
from multiprocessing import Pool

DATA_DIR = '../../input/shuffled_csv'


NUM_CLASS = 340
TRAIN_DF  = []
TEST_DF   = []

#print(os.listdir(DATA_DIR))



def read_one_df_file(df_file):
    """定义一个读取一个csv的函数，这个函数会当做参数传入并行处理的函数"""
    unused_cols = ["countrycode", "recognized", "timestamp", "cv"]
    name = df_file.split('_')[-2]
    print('%s ' % (name), end = ' ', flush=True)
    df = pd.read_csv(df_file)
    drop_cols = [col for col in unused_cols if col in df.columns]
    if len(drop_cols) > 0:
        df = df.drop(drop_cols, axis=1)
    return df


def multi_thread_read_df_files(df_files, processes=32):
    """并行读取多个csv"""
    start = dt.datetime.now()
    pool = Pool(processes=processes)
    dfs = pool.map(read_one_df_file, df_files)
    pool.close()
    pool.join()
    end = dt.datetime.now()
    print("\nTotal time:", (end - start).seconds, "seconds")
    
    big_df = pd.concat(dfs, ignore_index=False, sort=False)
    big_df.reset_index(drop=True, inplace=True)
    
    return big_df


def draw_cv2(raw_strokes, size=256, lw=5, time_color=True, base_size=256, point_drop_prob=0.0, 
             channel=1, stroke_drop_prob=0.0):
    """
    将一个涂鸦数据转换成一张黑百图, (size, size, channel)
    time_color: 若为true, 则每一笔颜色不一样,随着时间推移而变浅. 若为false,则颜色都为纯黑
    lw: 线段宽度
    point_drop_prob: 以此概率丢弃掉point, 0.05 - 0.15差不多
    channel: 必须为1或3

    """
    img = np.zeros((base_size, base_size, channel), np.uint8)
#     if channel == 1:
#         img = np.zeros((base_size, base_size), np.uint8)
    
#     elif channel == 3:
#         img = np.zeros((base_size, base_size, 3), np.uint8)
#     else:
#         assert False
    
    for t, stroke in enumerate(raw_strokes):
        if stroke_drop_prob > 0.0 and random.randint(0,100) < stroke_drop_prob * 100:
            continue
        for i in range(len(stroke[0]) - 1):
            if point_drop_prob > 0.0 and random.randint(0,100) < point_drop_prob * 100:
                continue
            color = 255 - min(t, 10) * 13 if time_color else 255
            if channel == 3:
                color = (color, color, color)
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != base_size:
        return cv2.resize(img, (size, size))    
    else:
        return img

def drawing_list_to_points(drawing_list, padding=True, max_len=500):
    """将csv中的drawing转换为一个numpy array
    padding: 若为true,则pad到max_len长
    max_len: 去掉长度超过max_len的点
    """
    strokes = []
    prex = drawing_list[0][0][0]
    prey = drawing_list[0][1][0]
    for i in range(len(drawing_list)):  # number of strokes
        for j in range(len(drawing_list[i][0])):  # number of points in one stroke
            strokes.append([drawing_list[i][0][j] - prex, drawing_list[i][1][j] - prey, 1, 0, 0])  # delta_x, delta_y
            prex = drawing_list[i][0][j]
            prey = drawing_list[i][1][j]
        strokes[-1][2] = 0
        strokes[-1][3] = 1  # end of stroke
    strokes[-1][3] = 0
    strokes[-1][4] = 1  # end of drawing
    strokes = np.array(strokes, dtype=np.float32) # shape:(point_num, 5)
    point_num = strokes.shape[0]
    
    if point_num > max_len:
        strokes = strokes[:max_len, :]
        strokes[-1][4] = 1  # end of drawing
        point_num = max_len
        return strokes, point_num  # point_num 是未padding的点数
    
    elif padding:
        padding_strokes = np.zeros((max_len, 5), dtype=np.float32)
        padding_strokes[:point_num, :] = strokes
        padding_strokes[point_num:, -1] = 1 # end of drawing
        return padding_strokes, point_num  # point_num 是未padding的点数

    

def drawing_list_to_array(drawing, size, channel, mode="train", point_drop_prob=0.0, stroke_drop_prob=0.0):
    assert mode in ["train", "eval"]
    if mode is "eval":
        point_drop_prob = 0
    image_array = np.zeros((size, size, channel))
#     image_array[:, :, :] = draw_cv2(drawing, size=size, point_drop_prob = point_drop_prob, channel=channel)
    if channel == 3:
        image_array[:, :, :] = draw_cv2(drawing, size=size, point_drop_prob = point_drop_prob, channel=channel,
                                        stroke_drop_prob=stroke_drop_prob)
    else:
        image_array[:, :, 0] = draw_cv2(drawing, size=size, point_drop_prob = point_drop_prob, channel=channel,
                                        stroke_drop_prob=stroke_drop_prob)
    
    return image_array    


def move_image(image, up, down, left, right): # 非负整数
    assert up * down == 0 and left * right == 0 # 不能同时左右移动,也不能同时上下移
    if up + down + left + right == 0:  # 不移动
        return image
    size = image.shape[0]
    up_pixel = size - up
    down_pixel = size - down
    left_pixel = size - left
    right_pixel = size - right

    moved_image = np.zeros((size, size, 3))
    moved_image[-down_pixel:up_pixel, -right_pixel:left_pixel, :] = image[-up_pixel:down_pixel, -left_pixel:right_pixel, :]
    return moved_image


class HorizonFlip(object):
    """
    水平翻转图片数组
    """
    def __init__(self, prob):
        assert prob >= 0.0 and prob <= 1.0
        self.prob = prob

    def __call__(self, image_array):
        """
        image_array: H X W X C
        """
        if self.prob > 0.0 and random.randint(0,100) <= self.prob * 100:
            return np.flip(image_array, 1).copy()
        return image_array

    
class RandomRightDownMove(object):
    """
    随机向右,下移动图片数组. 因为doodle数据是左上对齐的
    """
    def __init__(self, prob, move_range): # move_range是移动的pixel范围, 如(5, 10)
        assert prob >= 0.0 and prob <= 1.0
        self.prob = prob
        self.move_range = move_range

    def __call__(self, image_array):
        """
        image_array: H X W X C
        """
        down = 0
        right = 0
        if self.prob > 0.0:
            pixel = random.randint(self.move_range[0], self.move_range[1])
            if random.randint(0,100) <= self.prob * 100: # 下移pixel个像素
                down = pixel
                
            pixel = random.randint(self.move_range[0], self.move_range[1])
            if random.randint(0,100) <= self.prob * 100: # 右移pixel个像素
                right = pixel
                
            return move_image(image_array, 0, down, 0, right)
                            
        return image_array
    
    
class RandomMove(object):
    """
    随机移动图片数组
    """
    def __init__(self, prob, move_range): # move_range是移动的pixel范围, 如5(5, 10)
        assert prob >= 0.0 and prob <= 1.0
        self.prob = prob
        self.move_range = move_range

    def __call__(self, image_array):
        """
        image_array: H X W X C
        """
        up = 0
        down = 0
        left = 0
        right = 0
        if self.prob > 0.0 and random.randint(0,100) <= self.prob * 100:
            chance =  random.randint(0,100)
            pixel = random.randint(self.move_range[0], self.move_range[1])
            if chance < 33: # 上移pixel个像素
                up = pixel 
            elif chance < 67: # 下移
                down = pixel
                
            chance =  random.randint(0,100)
            pixel = random.randint(self.move_range[0], self.move_range[1])
            if chance < 33: # 左移pixel个像素
                left = pixel 
            elif chance < 67: # 右移
                right = pixel
                
            return move_image(image_array, up, down, left, right)
                            
        return image_array

    


class DoodleDataset(Dataset):
    def __init__(self, csv_files, channel, hps, transform, for_test=False, batch_num=None, for_cnn=True, for_rnn=False):
        super(DoodleDataset, self).__init__()
        self.channel = channel
        self.hps = hps
        self.transform = transform
        self.for_test = for_test
        self.batch_num=batch_num
        self.for_cnn = for_cnn
        self.for_rnn = for_rnn
        
        print("Load csv files:", end='')
        if len(csv_files) == 1:
            self.df = read_one_df_file(csv_files[0])
        else:
            self.df = multi_thread_read_df_files(csv_files)
        print("\nLoad done! info:")
        print(self.df.info())
        

    def __getitem__(self, index):
        drawing = self.df.loc[index,"drawing"]
        drawing_list = ast.literal_eval(drawing)
        
        image = None
        strokes = None
        point_num = None
        
        if self.for_cnn:
            image_array = drawing_list_to_array(drawing_list, size=self.hps.image_size, channel=self.channel,
                                                point_drop_prob=self.hps.point_drop_prob,
                                                stroke_drop_prob=self.hps.stroke_drop_prob)
            # PIL_image = Image.fromarray(image_array.astype('uint8'), 'RGB') # torchvision的一些transform要求输入是PIL图片
            #image = self.transform(PIL_image)
            image = self.transform(image_array)

            
        if self.for_rnn:
            strokes, point_num = drawing_list_to_points(drawing_list, padding=True, max_len=self.hps.max_seq_len)
            strokes = torch.from_numpy(strokes)
            # assert strokes.shape[0] == self.hps.max_seq_len 
            
        if self.for_cnn and self.for_rnn:
            sample = {'image':image, 'strokes':strokes, 'point_num':point_num}
        elif self.for_cnn:
            sample = {'image':image}
        else:
            sample = {'strokes':strokes, 'point_num':point_num}
            
        if not self.for_test:
            label = self.df.loc[index, "y"]
            label_name = self.df.loc[index,"word"]
            sample['label'] = label
            sample['label_name'] = label_name
            
        return sample

    def __len__(self):
        if self.batch_num is None:
            return len(self.df)
        return self.batch_num * self.hps.batch_size
    