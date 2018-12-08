# Image_classification
## 环境配置
- Python 3.6
- PyTorch >= 0.4.0

##数据集介绍
[CIFAR10](https://www.cnblogs.com/Jerry-Dong/p/8109938.html)

##训练
python main.py

##测试单张图片
python main.py  --test_only --pre_train ./pretrained_model/model_best.pt --data_test ./data/bird.jpg