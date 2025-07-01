# <div align='center'>基于深度学习的火灾检测与识别研究<div>

## 训练环境
windows11

python版本 3.8.2

pytorch版本 2.4.1

torchvision版本 0.20

ultralytics版本 8.3.28

CUDA版本 12.6

GPU RTX 3090（没有GPU可以调整各个train文件中的device参数，将device=[0,]修改为device='cpu')

## 训练准备
运行
```bash
pip install ultralytics
```
下载ultralytics

下载数据集，[数据集来源](https://aistudio.baidu.com/datasetdetail/166784)（点击跳转）

运行[VOCtoYOLO.py](VOCtoYOLO.py)得到dateset文件夹与labels文件夹

检查训练环境

下载YOLOv8.pt（所有[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时自动从最新的 Ultralytics [发布](https://github.com/ultralytics/assets/releases)下载）

准备[data.yaml](data.yaml)文件

编写并运行[train2.py](train2.py)得到普通训练结果

编写并运行[train4JianZhi.py](train4JianZhi.py)得到剪枝训练结果

编写并运行[train5LiangHua2.py](train5LiangHua2.py)得到量化训练结果

编写并运行[JianYuLiang.py](JianYuLiang.py)得到剪枝与量化训练结果

## 模型训练
该项目模型基于YOLOv8训练得出

普通训练结果保存在runs/train3/exp文件夹下

剪枝训练结果保存在runs/train2/exp3_pruned文件夹下

量化训练结果保存在runs/train2/exp_v3文件夹下

剪枝与量化训练结果保存在runs/train4/exp2文件夹下

runs文件夹暂时未上传


在所有文件夹中，模型统一保存在weights文件夹下

各个文件夹下都包含有F1曲线、混淆矩阵等图片



 
