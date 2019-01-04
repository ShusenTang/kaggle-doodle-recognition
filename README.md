# Quick, Draw! Doodle Recognition Challenge
本仓库是我在kaggle比赛[Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition)的所有代码（带注释），有兴趣的可以看看，顺便给个star。另外本次比赛的详细总结（包括大佬们的top方案分析)见可[我的博客](http://tangshusen.me/2018/12/05/kaggle-doodle-reco/#more)。    

这个比赛于12.05早上结束，最终结果是排在69/1316、铜牌，离银牌区差4名，还是比较遗憾的。不过这是我第一次花大量时间和精力在图像分类问题上，作为一个novice能拿到牌还算满意。

代码组织结构:
* `shuffle_csv.ipynb`对应的是博客2.1.1节，作用是将340个csv文件混合在一起然后再随机分成100份分别存储。
* `sketch_entropy.ipynb`对应博客2.1.3节，用熵（entropy）来分析数据中的outliers。（本次比赛未实际应用）
* `xception.py`对应博客2.2.1、2.2.2、2.2.3节，是本次比赛最后用的pytorch模型程序，依赖`data_loader`提供数据。
* `data_loader.py`对应博客2.1.2节，作用是为`xception.py`提供数据。
* `bug.ipynp`对应博客2.2.4节，是我在本次比赛遇到的一个教训。
* `rnn_model.py`为我写的简单的RNN模型，由于准确率不高所以最后并没有使用，也放在这里留作参考。
* `keras_model`文件夹为最开始使用的keras模型，后期未使用，留作参考。

另外输入文件组织如下:
```
- input
  - test_simplified.csv
  - sample_submission.csv
  - train_csv
    - airplane.csv
    - alarm clock.csv
    ...
  - shuffled_csv
    - train_100_1.csv.gz
    - train_100_2.csv.gz
    ...
    - train_100_100.csv.gz
```
其中shuffled_csv是经过`shuffle_csv.ipynb`处理得到的后续要使用的训练数据，其他的文件都可在[比赛数据页](https://www.kaggle.com/c/quickdraw-doodle-recognition/data)下载。
