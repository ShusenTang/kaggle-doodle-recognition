'''简单的RNN model,预测效果一般,所以后面并没有使用RNN'''
import os
import time
import ast
import argparse
import copy
import gc
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import DoodleDataset
from utils import *

from visdom import Visdom

viz = Visdom(env = "rnn_model")
assert viz.check_connection()

best_model_path = "./rnn_model/debug/csv71-90-epoch-03-step-008000-map3-0.799841.pth"  # 加载最好的模型继续训练或者进行测试

csv_files_path = '../../input/shuffled_csv'
model_dir = "./rnn_model"

NCATS=340
NCSVS=100
CHANNEL=3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if DEVICE == "cuda":
#     torch.cuda.set_device(device)


def get_HParams():
    parser = argparse.ArgumentParser(description='Get hyper params of the model.')
    parser.add_argument("-tag", type=str, default="debug") 
    parser.add_argument("-max_seq_len", type=int, default=500)
    parser.add_argument("-hidden_size", type=int, default=512)
    parser.add_argument("-rnn_model", type=str, default='lstm', help="The encoder RNN model, lstm or gru")
    parser.add_argument("-num_layers", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-dropout_prob", type=float, default=0.0)  # 根据torch.nn.LSTM, 当num_layers=1时, 这个参数无效
    
#   parser.add_argument("-image_size", type=int, default=224)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-gpus", type=int, required=True)
    
    hp = parser.parse_args()
    return hp


class RNN_Model(nn.Module):
    def __init__(self, hp, batch_first=True):
        super(RNN_Model, self).__init__()
        self.hp = hp
        self.batch_first = batch_first
        # 下面的全连接层是用encoder的输出h求decoder的初始h0和C0, 乘2是因为是双向的
        self.fc_output = nn.Linear(2 * hp.hidden_size, NCATS)
        nn.init.normal_(self.fc_output.weight, std=0.001)
        self.relu = nn.ReLU(inplace=True)

        # bidirectional RNN:
        if hp.rnn_model == 'lstm':
            self.rnn = nn.LSTM(input_size=5, hidden_size=hp.hidden_size, num_layers=hp.num_layers,
                               dropout=hp.dropout_prob, bidirectional=True, batch_first=batch_first)
        elif hp.rnn_model == 'gru':
            self.rnn = nn.GRU(input_size=5, hidden_size=hp.hidden_size, num_layers=hp.num_layers,
                              dropout=hp.dropout_prob, bidirectional=True, batch_first=batch_first)
        else:
            print("ERROR:RNN model must be lstm or gru!!!")
            assert False

        # 激活 dropout:
        self.train()

    def feature(self, inputs, act_lens, h=None, c=None):
        """
        :param inputs shape (batch_first): (batch_size, seq_len, 5) or (not batch_first) (seq_len, batch_size, 5)
        :param h:
        :param c:
        :return: output, shape: (batch_size, seq_len, 2*enc_size)
        """
#         if h is None:
#             # then must init with zeros
#             h = torch.zeros(self.hp.batch_size, 2 * self.hp.num_layers,  self.hp.hidden_size, device=DEVICE)
#             c = torch.zeros(self.hp.batch_size, 2 * self.hp.num_layers, self.hp.hidden_size, device=DEVICE)

        if self.hp.rnn_model == 'lstm':
            # output包含了最后一层lstm的所有时间步的hidden_state, 而h和c分别包含了所有层的最后一个时间步的hidden_state和cell_state
            output, (h, c) = self.rnn(inputs.float())
        elif self.hp.rnn_model == 'gru':
            # output包含了最后一层gru的所有时间步的hidden_state, 而h包含了所有层的最后一个时间步的hidden_state
            output, h = self.rnn(inputs.float())
        else:
            assert False

        # output shape: (seq_len, batch_size, 2*hidden_size)
        # output = output.permute(1, 0, 2)  # 现在output shape: (batch_size, seq_len, 2*hidden_size)
        act_lens = act_lens - 1  # index 是从0开始的
        indices = act_lens.view(-1, 1, 1).repeat(1, 1, output.shape[-1]) # shape: (batch, 1, 2*hidden_size)
        act_last_h = torch.gather(output, 1, indices) # 取实际的最后一个h, shape: (batch, 1, 2*hidden_size)
        act_last_h = torch.squeeze(act_last_h, 1)  # shape: (batch, 2*hidden_size)
        
        return act_last_h
        
    
    def forward(self, inputs, act_lens, h=None, c=None):
        act_last_h = self.feature(inputs, act_lens, h, c)
        x = self.fc_output(act_last_h)
        logit = self.relu(x)

        # logit shape: (batch_size, 340)
        return logit


    
def plot_by_visdom(win, x, y, opts):
    if win is None:
        win = viz.line(X = x, Y = y, opts=opts)
    else:
        viz.line(X= x, Y = y, win=win, update='append', opts=opts)
    
    return win
    
    
def valid_model(model, valid_data_loader):
    model.eval()   # Set model to   mode
    
    correct_num = 0
    map3s = []
    top3_accs = []
    for samples_batch in tqdm(valid_data_loader):
        inputs = samples_batch["strokes"]
        labels = samples_batch["label"]
        act_lens = samples_batch["point_num"]
        
        inputs = inputs.to(DEVICE, dtype=torch.float)
        act_lens = act_lens.to(DEVICE, dtype=torch.long)
        labels = labels.to(DEVICE, dtype=torch.long)
        
        with torch.set_grad_enabled(False):
            # forward
            outputs = model(inputs, act_lens)
            top3_acc, map3 = mapk(outputs, labels, k=3)
            map3s.append(map3)
            top3_accs.append(top3_acc)

            _, preds = torch.max(outputs, 1)
            correct_num += torch.sum(preds.int() == labels.int())
    
    valid_acc = correct_num.double().item() / len(valid_data_loader.dataset)
    valid_map3 = sum(map3s) / float(len(map3s))
    valid_top3_acc = sum(top3_accs) / float(len(top3_accs))

    model.train()  # Set model to training mode
    
    return valid_acc, valid_map3, valid_top3_acc
     
    
def train_model(hps, model, train_data_loader, valid_data_loader, criterion, optimizer, num_epochs=20, 
                print_every_step=20, save_every_step=None, batch_accumulate_size=1):
    train_start = time.time()
    train_steps = int(np.ceil(len(train_data_loader.dataset) / float(hps.batch_size)))
    if save_every_step is None:
        save_every_step = int(train_steps / 10)  # 1/10个epoch检查(保存)一次
    
    model = model.to(DEVICE) # to gpu
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_valid_map3 = 0.0
    best_valid_map3 = float("0." + best_model_path.split(".")[-2])
    
    valid_win = None
    valid_opts = dict(title="valid", xlabel="step", ylabel="acc or map3", 
                          legend=['valid_acc', "valid_map3", "valid_top3_acc"])

    step_count = 0
    for epoch in range(num_epochs):
        print('Epoch %d/%d' % (epoch+1, num_epochs))
        print('-' * 20)
        model.train()  # Set model to training mode
        
        # 每个epoch一个图
        train_win = None
        train_opts = dict(title="train-epoch-%d"%(epoch+1), xlabel="step", ylabel="acc or loss", 
                          legend=['train_acc', "train_loss"])


        # Iterate over data.
        for step, samples_batch in enumerate(train_data_loader):
            start = time.time()
            inputs = samples_batch["strokes"]
            act_lens = samples_batch["point_num"]
            labels = samples_batch["label"]
            
            
            # loss, preds, step_acc = train_step(model, inputs, labels, criterion, optimizer)
            # 上面一行代码就是普通的训练过程,下面这部分使用了梯度累加的trick,相当于增大了batch_size
            # 参考https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822中Gopal_Sharma的系列回答
            #################将batch_accumulate_size个batch的梯度积累起来,只在最后一次更新网络参数###################
            inputs = inputs.to(DEVICE, dtype=torch.float)
            act_lens = act_lens.to(DEVICE, dtype=torch.long)
            labels = labels.to(DEVICE, dtype=torch.float)
            if step % batch_accumulate_size == 0: 
                optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs, act_lens)
                loss = criterion(outputs, labels.long()) / batch_accumulate_size # 一定要除以这个size,原因见上面链接的讨论
                loss.backward()

                _, preds = torch.max(outputs, 1)
            correct_num = torch.sum(preds == labels.long())
            step_acc = correct_num.double() / inputs.size(0) 
                
            if (step + 1) % batch_accumulate_size == 0:
                optimizer.step()
            
            loss = batch_accumulate_size * loss.item() # 转换为数字方便后面用visdom画图
            step_acc = step_acc.item()
            ########################################################################################
                
            
            train_win = plot_by_visdom(train_win, np.array([step_count]), 
                                       np.column_stack((np.array([step_acc]), np.array([loss]))), train_opts)


            if step % print_every_step == 0:
                print('Train RNN epoch %d/%d step %d/%d. Loss: %.4f Acc: %.4f.  time taken: %.2fs' % (
                    epoch+1, num_epochs, step, train_steps, loss, step_acc, time.time()-start))

            if (step > 0 and step % save_every_step == 0) or (step == train_steps - 1):
                valid_acc, valid_map3, valid_top3_acc = valid_model(model, valid_data_loader)
                scheduler.step(valid_map3) # 适当调整学习率
                print('Validating model ...\nbest_valid_map3:%.6f valid_acc:%.4f valid_map3:%.6f valid_top3_acc:%.6f' % (
                        best_valid_map3, valid_acc, valid_map3, valid_top3_acc))
                
                valid_win = plot_by_visdom(valid_win, np.array([step_count]), 
                                           np.column_stack((np.array([valid_acc]), 
                                                            np.array([valid_map3]), 
                                                            np.array([valid_top3_acc]))), 
                                           valid_opts)
                
                if best_valid_map3 < valid_map3:
                    best_valid_map3 = valid_map3
                    print("Better model. best_valid_map3 --> %.6f \nsaving model..." % best_valid_map3, end='')
                    save_name = "csv71-90-epoch-%02d-step-%06d-map3-%.6f.pth" % (epoch + 1, step, best_valid_map3)
                    save_model(model, model_save_dir=model_dir + "/" + hps.tag, save_name=save_name)
            
            del inputs, labels, outputs
            gc.collect()
            step_count += 1                
    
    # valid_acc, valid_map3, valid_top3_acc = valid_model(model, valid_data_loader)
    save_name = "whole_model_epoch_%d.pth" % num_epochs
    save_model(model, model_save_dir=model_dir + "/" + hps.tag, save_name=save_name)

    time_elapsed = time.time() - train_start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid map3:%.6f' % best_valid_map3) 
    
    
def main():
    hps = get_HParams()
    
    hps_dict = vars(hps)
    for key, val in hps_dict.items():
        print('%s = %s' % (key, str(val)))
    
    # train_csv_files, valid_csv_files = get_csv_files_randomly(hps.train_files_num, hps.valid_files_num)
    train_ks = list(range(71, 91))
    if 12 in train_ks:
        train_ks.remove(12)  # 12是验证集
    assert 12 not in train_ks
    train_csv_files = [os.path.join(csv_files_path, 'train_%d_%d.csv.gz'%(k, NCSVS)) for k in train_ks]
    valid_csv_files = [os.path.join(csv_files_path, 'train_12_100.csv.gz')]
    
    model = RNN_Model(hp=hps)
    model.load_state_dict(torch.load(best_model_path)) # 将之前训练的模型load进来
    
    if hps.gpus > 1:
        print("------------use %d GPUs!------------" % hps.gpus)
        model = nn.DataParallel(model)
    # 如果使用了DataParallel,那么load也应该在它后面load,否则会报key对不上的错, 参考https://www.ptorch.com/news/74.html
    # model.load_state_dict(torch.load(best_model_path)) # 将之前训练的模型load进来

    train_data_loader = create_data_loader(csv_files=train_csv_files, channel=CHANNEL, hps=hps,
                                           transform=None, batch_num=None, for_cnn=False, for_rnn=True)
    valid_data_loader = create_data_loader(csv_files=valid_csv_files, channel=CHANNEL, hps=hps,
                                           transform=None, batch_num=400, for_cnn=False, for_rnn=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.learning_rate, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    
    train_model(hps, model, train_data_loader, valid_data_loader, criterion, optimizer, 
                              num_epochs=hps.epochs, print_every_step=400, save_every_step=4000, batch_accumulate_size=8)
    
    
    
if __name__ == '__main__':
    main()

