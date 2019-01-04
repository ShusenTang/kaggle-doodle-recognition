import os
import datetime as dt
import gc
import ast
import argparse

import cv2
import pandas as pd
import numpy as np

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam

from keras.utils import multi_gpu_model

from keras.applications import ResNet50
# from tensorflow.keras.applications import MobileNet

#from tensorflow.keras.applications.mobilenet import preprocess_input # mobilenet输入预处理, 将 0-255 处理成 -1 ~ 1
from keras.applications.resnet50 import preprocess_input

from utils import *

start = dt.datetime.now()

# 100个csv文件行数: [496187, 498017, 496445, 496510, 497595, 495627, 497747, 495595, 497669, 497038, 497073, 497010, 496040, 497867, 497248, 498626, 496410, 497933, 497850, 497071, 496888, 497305, 498025, 495586, 497263, 496805, 497485, 496926, 496603, 499216, 496182, 497092, 497498, 497154, 497438, 496961, 498351, 495871, 497188, 497187, 495829, 497037, 497616, 497683, 494729, 497923, 496708, 496374, 497231, 498460, 498786, 496202, 497518, 496625, 496419, 497551, 497014, 497653, 495304, 497496, 496799, 495842, 496695, 497414, 497925, 496472, 497099, 497466, 496377, 496823, 498418, 496797, 496807, 497284, 498669, 496705, 497727, 496606, 497032, 497060, 497505, 497177, 495795, 496996, 496764, 497767, 496167, 497157, 498249, 496523, 498350, 495811, 498178, 496174, 497469, 497085, 496558, 497822, 495535, 497740]
# 最大: 499216
# 最小: 494729

DP_DIR = '../input/shuffled_csv'  # 共100个shuffled文件, 每一个文件有 494000+ 个sample, 所有共 4790w+ 个sample
INPUT_DIR = '../input'
SUBMIT_DIR = '../submission/ResNet50'
LOG_DIR = '../log/ResNet50'
MODEL_DIR = '../models/ResNet50'

check_dirs([SUBMIT_DIR, LOG_DIR, MODEL_DIR])


ROWS_PER_FILE = 494729

BASE_SIZE = 256 # 原始数据范围0-256, 参见https://github.com/googlecreativelab/quickdraw-dataset#the-raw-moderated-dataset

NCSVS = 100
NCATS = 340
np.random.seed(seed=1995)
#tf.set_random_seed(seed=1995)

def get_HParams():
    parser = argparse.ArgumentParser(description='Get hyper params of the model.')
    parser.add_argument("-tag", type=str, default="debug") 
    parser.add_argument("-gpus", type=int, required=True)
    parser.add_argument("-image_size", type=int, default=224)
    parser.add_argument("-rgb", type=ast.literal_eval, default=True) # channel = 1 or 3

    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-batch_size", type=int, default=256)
    parser.add_argument("-dataset_prob", type=float, default=1.0) # 用多少比例数据进行训练 0.0 ~ 1.0

    
    # 随机选择train和valid文件
    parser.add_argument("-train_files_num", type=int, default=99)  # train和valid一共最多100个file
    parser.add_argument("-valid_files_num", type=int, default=1)
    parser.add_argument("-eval_samples_num", type=int, default=34000) # 训练完成后用 valid的一部分 进行评估得到map3分数

    # 数据增强
    parser.add_argument("-point_drop_prob", type=float, default=0.1) # 随机丢弃数据点的概率
    
    parser.add_argument("-debug", type=ast.literal_eval, default=False)
    parser.add_argument("-use_pretrain_model", type=ast.literal_eval, default=False)


    hps = parser.parse_args()
    return hps

hps = get_HParams()
print_save_HParams(hps, os.path.join(LOG_DIR, "HParams_" + hps.tag + ".json"))

BATCHES_PER_FILE = int(ROWS_PER_FILE * hps.dataset_prob) // hps.batch_size
STEPS = hps.train_files_num * BATCHES_PER_FILE  
valid_STEPS = hps.valid_files_num * BATCHES_PER_FILE
test_STEPS = np.ceil(112163.0 / hps.batch_size)

CHANNEL = 3 if hps.rgb is True else 1

if hps.debug:
    hps.epochs = 2
    hps.eval_samples_num = 3400
    STEPS = 500 # debug的时候steps可以设置小一点
    valid_STEPS= 50
    print("\n---------debug mode!---------\n")
    
print("EPOCHS = %d \nSTEPS = %d \nvalid_STEPS = %d " % (hps.epochs, STEPS, valid_STEPS))


if hps.use_pretrain_model:
    pretrain_model = ResNet50(weights="imagenet", include_top=False)
    x = pretrain_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predict = Dense(NCATS, activation='softmax')(x)
    single_model = Model(inputs=pretrain_model.input, outputs=predict)
else:
    single_model = ResNet50(input_shape=(hps.image_size, hps.image_size, CHANNEL), weights=None, classes=NCATS)


print("\n ---------------Using %d GPUs----------------\n" % hps.gpus)
model = multi_gpu_model(single_model, gpus=hps.gpus)

model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())

random_ks = np.random.permutation([i+1 for i in range(NCSVS)])
#train_data_gen = image_generator_xd(hps.image_size, hps.batch_size, random_ks[:hps.train_files_num], preprocess_input,
#                                    point_drop_prob=hps.point_drop_prob, # 数据增强
#                                    channel=CHANNEL)
#valid_data_gen = image_generator_xd(hps.image_size, hps.batch_size, random_ks[-hps.valid_files_num:], preprocess_input,
#                                    point_drop_prob=0.0, # 不数据增强
#                                    channel=CHANNEL)
train_data_gen = Iamge_Sequence(hps.image_size, hps.batch_size, random_ks[:hps.train_files_num], BATCHES_PER_FILE, 
                                preprocess_input, point_drop_prob=hps.point_drop_prob, # 数据增强
                                channel=CHANNEL)
valid_data_gen = Iamge_Sequence(hps.image_size, hps.batch_size, random_ks[-hps.valid_files_num:], BATCHES_PER_FILE,
                                preprocess_input, point_drop_prob=0.0, # 不数据增强
                                channel=CHANNEL)


weight_path =  os.path.join(MODEL_DIR, "best_weight_" + hps.tag + ".hdf5")
callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
                      min_delta=0.005, mode='max', cooldown=3, verbose=1),
    CSVLogger(os.path.join(LOG_DIR, "log_" + hps.tag + ".csv")),
    EarlyStopping(monitor='val_categorical_accuracy', patience=15, mode='max'),
    ModelCheckpoint(weight_path, monitor='val_categorical_accuracy', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)
    ]

hists = []
hist = model.fit_generator(
    generator=train_data_gen, steps_per_epoch=STEPS, epochs=hps.epochs, verbose=1,
    validation_data=valid_data_gen, validation_steps=valid_STEPS,
    callbacks = callbacks,
    
    workers=16,
    use_multiprocessing=True # 设置为true的话会变慢而且报warning,说是会有数据复制
)

# 为了计算map3的值
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_%d_%d.csv.gz'%(random_ks[-1], NCSVS)), nrows=hps.eval_samples_num)
x_valid = df_to_gray_image_array(valid_df, hps.image_size, preprocess_input,
                                 point_drop_prob=0.0, # 不数据增强
                                 channel=CHANNEL)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Evaluation data array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))
valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.6f}'.format(map3))


if not hps.debug:
    # submit
    test_start = dt.datetime.now()
    print("testing......")
#     test_df_path = os.path.join(INPUT_DIR, 'test_simplified.csv')
#     test_data_gen = test_image_generator(test_df_path, hps.image_size, hps.batch_size, preprocess_input,
#                                          point_drop_prob=0, # 要数据增强吗
#                                          channel=CHANNEL)

#     test_predictions = model.predict_generator(test_data_gen, steps=test_STEPS, verbose=1)

    test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
    x_test = df_to_gray_image_array(test, hps.image_size, preprocess_input,
                                    point_drop_prob=0, # 要数据增强吗
                                    channel=CHANNEL)
    test_predictions = model.predict(x_test, batch_size=hps.batch_size, verbose=1)
    

    top3 = preds2catids(test_predictions)

    cats = list_all_categories()
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
    top3cats = top3.replace(id2cat)

    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = test[['key_id', 'word']]
    submission.to_csv(os.path.join(SUBMIT_DIR, "ResNet50_" + hps.tag + "_%05d.csv" % int(map3 * 10**6) ), index=False)
    
    print("Model test1 total time:".format((dt.datetime.now() - test_start).seconds))

    ################ues best model to predict#########
    test_start = dt.datetime.now()
    print("Evaluating best model......")
    model.load_weights(weight_path)

    valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
    map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
    print('Map3: {:.6f}'.format(map3))

    # submit
    #test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
    x_test = df_to_gray_image_array(test, hps.image_size, preprocess_input,
                                    point_drop_prob=0, # 要数据增强吗
                                    channel=CHANNEL)
    test_predictions = model.predict(x_test, batch_size=hps.batch_size, verbose=1)

    top3 = preds2catids(test_predictions)

    cats = list_all_categories()
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
    top3cats = top3.replace(id2cat)

    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = test[['key_id', 'word']]
    submission.to_csv(os.path.join(SUBMIT_DIR, "ResNet50_" + hps.tag + "_best_%05d.csv" % int(map3 * 10**6) ), index=False)
    print("Model test2 total time:".format((dt.datetime.now() - test_start).seconds))


end = dt.datetime.now()
print('ResNet50 model latest run {}.\nTotal time {} s'.format(end, (end - start).seconds))
