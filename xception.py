'''能拿到73名的我的最好模型(PB0.94185)'''
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
from collections import OrderedDict

import torch
import torch.nn as nn
import pretrainedmodels
from pretrainedmodels.models.xception import Xception

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from visdom import Visdom


from data_loader import DoodleDataset, HorizonFlip, RandomMove, RandomRightDownMove
from utils import preds2catids, list_all_categories


TEST_MODEL = False

if not TEST_MODEL:
    viz = Visdom(env = "xception")
    assert viz.check_connection()

xception_path = "/S1/CSCL/tangss/pretrained_models/pytorch/xception-43020ad28.pth"
best_model_path = "./models/1123/csv81-100-epoch-01-step-060000-map3-0.907089.pth" # 加载最好的模型继续训练或者进行测试

csv_files_path = '../../input/shuffled_csv'
INPUT_DIR = "../../input"
model_dir = "./models"
submit_dir = "./submission"

NCATS=340
NCSVS=100
CHANNEL=3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_HParams():
    parser = argparse.ArgumentParser(description='Get hyper params of the model.')
    parser.add_argument("-tag", type=str, default="debug") 
    parser.add_argument("-image_size", type=int, default=299)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-gpus", type=int, required=True)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-learning_rate", type=float, default=0.001)

    parser.add_argument("-dataset_prob", type=float, default=1.0) # 用多少比例数据进行训练 0.0 ~ 1.0

    
    # 随机选择train和valid文件
    parser.add_argument("-train_files_num", type=int, default=99)  # train和valid一共最多100个file
    parser.add_argument("-valid_files_num", type=int, default=1)
    parser.add_argument("-eval_samples_num", type=int, default=34000) # 训练完成后用 valid的一部分 进行评估得到map3分数

    parser.add_argument("-point_drop_prob", type=float, default=0.0)
    parser.add_argument("-stroke_drop_prob", type=float, default=0.0)

    
    #parser.add_argument("-debug", type=ast.literal_eval, default=True)

    hps = parser.parse_args()
    return hps


def create_model(num_classes=NCATS, model_func=Xception, pretrained_path=xception_path):
    # model_name = 'xception'
    # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = model_func(num_classes=1000)
    model.load_state_dict(torch.load(pretrained_path))
    fc_in_feas = model.fc.in_features
    model.fc = nn.Linear(fc_in_feas, num_classes)
    model.last_linear = model.fc
    return model


def save_model(model, model_save_dir, save_name):
    """
    save_name : "xxxx.pth"
    """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model.state_dict(), os.path.join(model_save_dir, save_name))
    print("save done!")


def create_data_loader(csv_files, channel, hps, transform, for_test=False, batch_num=None):
    dataset = DoodleDataset(csv_files=csv_files, channel=channel, hps=hps, transform=transform, 
                            for_test=for_test, batch_num=batch_num)
    
    shuffle = False if for_test else True
    drop_last = False if for_test else True
    #batch_size = 64 if for_test else hps.batch_size
    dataloader = DataLoader(dataset, batch_size=hps.batch_size, shuffle=shuffle, num_workers=16,
                            pin_memory=True,drop_last=drop_last)
    return dataloader


def get_csv_files_randomly(train_files_num, valid_files_num):
    assert train_files_num + valid_files_num <= 100
    random_ks = np.random.permutation([i+1 for i in range(NCSVS)])
    train_ks = random_ks[:train_files_num]
    valid_ks = random_ks[-valid_files_num:]
    train_files = [os.path.join(csv_files_path, 'train_%d_%d.csv.gz'%(k, NCSVS)) for k in train_ks]
    valid_files = [os.path.join(csv_files_path, 'train_%d_%d.csv.gz'%(k, NCSVS)) for k in valid_ks]
    return train_files, valid_files
 
    
def plot_by_visdom(win, x, y, opts):
    if win is None:
        win = viz.line(X = x, Y = y, opts=opts)
    else:
        viz.line(X= x, Y = y, win=win, update='append', opts=opts)
    
    return win
    
    
def mapk(output, target, k=3):
    """
    Computes the mean average precision at k.
    
    Parameters
    ----------
    output (torch.Tensor): A Tensor of predicted elements.
                           Shape: (N,C)  where C = number of classes, N = batch size
    target (torch.int): A Tensor of elements that are to be predicted. 
                        Shape: (N) where each value is  0≤targets[i]≤C−1
    k (int, optional): The maximum number of predicted elements
    
    Returns
    -------
    mapk_score (torch.float):  The mean average precision at k over the output
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k=k, dim=1, largest=True, sorted=True)
        pred = pred.t()  # 3 X batch_size
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        topk_acc = torch.sum(correct.float()) / batch_size

        for i in range(k):
            correct[i] = correct[i]*(k-i)
            
        mapk_score = correct[:k].view(-1).float().sum(0, keepdim=True)
        mapk_score.mul_(1.0 / (k * batch_size))
        return topk_acc, mapk_score

    
def test_model(model, dataloaders, submit_name):
    """测试模型并生成提交csv
    dataloaders是包含多个dataloader的列表,最终结果是取预测平均值
    """  
    model = model.to(DEVICE) # to gpu
    model.eval()  
    test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
    avg_preds = np.zeros((test.shape[0], NCATS))
    
    with torch.set_grad_enabled(False):
        for dataloader in dataloaders:
            preds = np.empty((0, NCATS))
            for sample_batch in tqdm(dataloader):
                x = sample_batch["image"].to(DEVICE, dtype=torch.float)
                output = model(x)
                
                preds = np.concatenate([preds, output], axis = 0)

            avg_preds += preds
        
    avg_preds /= len(dataloaders)
    
    top3 = preds2catids(avg_preds)
    cats = list_all_categories()
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
    top3cats = top3.replace(id2cat)
    
    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = test[['key_id', 'word']]
    submission.to_csv(os.path.join(submit_dir, submit_name), index=False)
        

def valid_model(model, valid_data_loader):
    model.eval()   # Set model to   mode
    
    correct_num = 0
    map3s = []
    top3_accs = []
    for samples_batch in tqdm(valid_data_loader):
        inputs = samples_batch["image"]
        labels = samples_batch["label"]
        inputs = inputs.to(DEVICE, dtype=torch.float)
        labels = labels.to(DEVICE, dtype=torch.long)
        
        with torch.set_grad_enabled(False):
            # forward
            outputs = model(inputs)
            top3_acc, map3 = mapk(outputs, labels, k=3)
            map3s.append(map3.item())
            top3_accs.append(top3_acc.item())

            _, preds = torch.max(outputs, 1)
            correct_num += torch.sum(preds.int() == labels.int())
    
    valid_acc = correct_num.double().item() / len(valid_data_loader.dataset)
    valid_map3 = sum(map3s) / float(len(map3s))
    valid_top3_acc = sum(top3_accs) / float(len(top3_accs))

    model.train()  # Set model to training mode
    
    return valid_acc, valid_map3, valid_top3_acc


    
def train_step(model, inputs, labels, criterion, optimizer):
    inputs = inputs.to(DEVICE, dtype=torch.float)
    labels = labels.to(DEVICE, dtype=torch.float)

    # zero the parameter gradients
    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())

        _, preds = torch.max(outputs, 1)

        # backward + optimize
        loss.backward()
        optimizer.step()
    
    correct_num = torch.sum(preds == labels.long())
    step_acc = correct_num.double() / inputs.size(0)
    
    return loss.item(), preds, step_acc.item()
    
    
    
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
            inputs = samples_batch["image"]
            labels = samples_batch["label"]
            
            
            # loss, preds, step_acc = train_step(model, inputs, labels, criterion, optimizer)
            # 上面一行代码就是普通的训练过程,下面这部分使用了梯度累加的trick,相当于增大了batch_size
            # 参考https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822中Gopal_Sharma的系列回答
            #################将batch_accumulate_size个batch的梯度积累起来,只在最后一次更新网络参数###################
            inputs = inputs.to(DEVICE, dtype=torch.float)
            labels = labels.to(DEVICE, dtype=torch.float)
            if step % batch_accumulate_size == 0: 
                optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs)
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
                print('Train epoch %d/%d step %d/%d. Loss: %.4f Acc: %.4f.  time taken: %.2fs' % (
                    epoch+1, num_epochs, step, train_steps, loss, step_acc, time.time()-start))

            if (step > 0 and step % save_every_step == 0) or (step == train_steps - 2):
                valid_acc, valid_map3, valid_top3_acc = valid_model(model, valid_data_loader)
                scheduler.step(valid_map3) # 适当调整学习率
                print('Validating model ...\n best_valid_map3:%.6f valid_acc:%.4f valid_map3:%.6f valid_top3_acc:%.6f' % (
                        best_valid_map3, valid_acc, valid_map3, valid_top3_acc))
                
                valid_win = plot_by_visdom(valid_win, np.array([step_count]), 
                                           np.column_stack((np.array([valid_acc]), 
                                                            np.array([valid_map3]), 
                                                            np.array([valid_top3_acc]))), 
                                           valid_opts)
                
                if best_valid_map3 < valid_map3:
                    best_valid_map3 = valid_map3
                    print("Better model. best_valid_map3 --> %.6f \nsaving model..." % best_valid_map3, end='')
                    save_name = "csv81-100-epoch-%02d-step-%06d-map3-%.6f.pth" % (epoch + 1, step, best_valid_map3)
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
    train_ks = list(range(81, 101))
    if 12 in train_ks:
        train_ks.remove(12)  # 12是验证集
    assert 12 not in train_ks
    train_csv_files = [os.path.join(csv_files_path, 'train_%d_%d.csv.gz'%(k, NCSVS)) for k in train_ks]
    valid_csv_files = [os.path.join(csv_files_path, 'train_12_100.csv.gz')]
    
    model = create_model()
    if hps.gpus > 1:
        print("------------use %d GPUs!------------" % hps.gpus)
        model = nn.DataParallel(model)
    # 如果使用了DataParallel,那么load也应该在它后面load,否则会报key对不上的错, 参考https://www.ptorch.com/news/74.html
     
    pretrained_net_dict = torch.load(best_model_path)
    if hps.gpus == 1 and list(pretrained_net_dict.keys())[0][:6] == "module":
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(pretrained_net_dict) # 将之前训练的模型load进来

    if not TEST_MODEL:
        train_data_transforms = transforms.Compose([#transforms.RandomRotation(degrees=15),
                                              #transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5] ), # 需要和pretrainedmodels要求的一样
                                            ])
        valid_data_transforms = transforms.Compose([#transforms.RandomRotation(degrees=15),
                                              #transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                       std=[0.5, 0.5, 0.5] ), # 需要和pretrainedmodels要求的一样
                                            ])
        train_data_loader = create_data_loader(csv_files=train_csv_files, channel=CHANNEL, hps=hps,
                                               transform=train_data_transforms, batch_num=None)
        valid_data_loader = create_data_loader(csv_files=valid_csv_files, channel=CHANNEL, hps=hps,
                                               transform=valid_data_transforms, batch_num=1000)

        optimizer = torch.optim.Adam(model.parameters(), lr=hps.learning_rate, amsgrad=True)
        criterion = nn.CrossEntropyLoss()

        train_model(hps, model, train_data_loader, valid_data_loader, criterion, optimizer, 
                                  num_epochs=hps.epochs, print_every_step=1000, save_every_step=10000, batch_accumulate_size=8)
        
    else:
        print("-----test model !-----")
        test_csv_file = os.path.join(INPUT_DIR, 'test_simplified.csv')
        #test_csv_file = os.path.join(csv_files_path, 'train_99_100.csv.gz')
        
        test_transform = []
        test_transform_1 = transforms.Compose([#transforms.RandomRotation(degrees=15),
                                              #transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5] ), # 需要和pretrainedmodels要求的一样
                                            ])
        test_transform.append(test_transform_1)

        
        # 水平翻转
        test_transform_2 = transforms.Compose([HorizonFlip(prob=1.0),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5] ), # 需要和pretrainedmodels要求的一样
                                            ])
        test_transform.append(test_transform_2)
               

#         test_transform_3 = transforms.Compose([RandomRightDownMove(prob=1.0, move_range=(5, 20)),
#                                               transforms.ToTensor(),
#                                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                                                    std=[0.5, 0.5, 0.5] ), # 需要和pretrainedmodels要求的一样
#                                             ])
#         test_transform.append(test_transform_3)
        
        
#         test_transform_4 = transforms.Compose([RandomMove(prob=1.0, move_range=(15, 25)),
#                                               transforms.ToTensor(),
#                                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                                                    std=[0.5, 0.5, 0.5] ), # 需要和pretrainedmodels要求的一样
#                                             ])
#         test_transform.append(test_transform_4)
        
#         test_transform_5 = transforms.Compose([RandomMove(prob=1.0, move_range=(20, 30)),
#                                               transforms.ToTensor(),
#                                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                                                    std=[0.5, 0.5, 0.5] ), # 需要和pretrainedmodels要求的一样
#                                             ])
#         test_transform.append(test_transform_5)

     
        dataloaders = []
        for trans in test_transform:
            tmp_loader = create_data_loader(csv_files=[test_csv_file], channel=CHANNEL, hps=hps,
                                               transform=trans, 
                                                for_test=True,
                                             )
            dataloaders.append(tmp_loader)
            
        
        # dataloaders = [test_data_loader_2, test_data_loader_1]
        print(len(dataloaders), "dataloader!")
        submit_name = "xception_map3_" + best_model_path.split(".")[-2] + "_aug_2.csv"
        test_model(model, dataloaders, submit_name)


if __name__ == '__main__':
    main()
