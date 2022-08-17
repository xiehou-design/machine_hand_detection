# 机械手势识别

本项目一共提供了两套手势识别数据集，一套是基于网上的偏卡通的手势数据集，另外一套是自己采集的机械手手势数据。第一套数据集存放在`./Hand/hand1`文件夹下，第二套数据集存放在`./Hand/hand2`文件夹下。

## 数据处理

数据处理包含数据集主要包含数据集划分和自定义数据加载`Dataset`类。

第一套数据集划分使用`./Hand/data/data_split.py`代码，第二套数据集划分采用`./Hand/data/data_split_hand2.py`代码。执行这个两个代码中的一个，将会在同级目录下生成`train.txt`和`test.txt`两个文本文件。

数据加载类分别写在`./Hand/model_hand_1/dataset.py`和`./Hand/model_hand_2/dataset.py`可以按照自己的需求来更改数据加载方式。

当前的图像训练都是将图像放缩到`100 x 100`的大小，可以自定义调整，调整方式比较简单。

## 模型构建

目前两套数据集共用一个模型，也就是`./Hand/model_hand_1/model.py`和`./Hand/model_hand_2/model.py`是一样的。为了方便更高分辨率图像的训练和预测，在`./Hand/model_hand_2/model.py`中提供了**ResNet50**的模型，可以便于使用更加深层模型。如果将图像放缩到`224 x 224`大小的话，可以使用该程序下面的预训练模型权重进行训练，能更快加速模型的收敛速度。

预训练模型调用加载如下所示：

```python
resModel = ResModel(classes=6)
new_state = {}
weight_state = torch.load('./pretrained/resnet50-19c8e357.pth')
for key, value in weight_state.items():
    if 'fc' not in key:
        new_state['model.' + key] = value
    else:
        new_state['model.' + key] = resModel.state_dict()['model.' + key]
resModel.load_state_dict(new_state)
```

## 模型训练

模型训练的话可以直接执行`./Hand/model_hand_1/train.py`或`./Hand/model_hand_2/train.py`文件，这两个训练文件分别是训练第一套数据集或第二套数据集，但一定要注意的是`./Hand/data/train.txt`和`./Hand/data/train.txt`是否和数据集对应上，如果没对应上则可能出现数据集无法正确加载错误。

两个`train.py`文件主要需要修改的超参数包含如下：

```python
train_transform = transforms.Compose([
    transforms.RandomRotation((-30, 30)),
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop((100, 100)),
    transforms.ToTensor()
])
test_transformer = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.CenterCrop((100, 100)),
    transforms.ToTensor()
])

batch_size = 8
learning_rate = 1e-3
epochs = 100
```

数据增强的旋转方式、图像放缩的尺度变化、图像剪裁的尺度、`batch_size`超参数的调整、`learning_rate`超参数的调整、`epochs`超参数的调整。

## 软件使用

在这个项目中，我们也制作了一个可视化的软件，便于项目成果的演示，主要包含这几个文件，具体如下下图1所示：

![image-20220817101914776](D:\Python\project\machine_hand_detection\pictures\image-20220817101914776.png)

<center>图1 软件包含的代码文件</center>

主要需要保留的有三个文件：`main.py`、`ui.py`和`savemodel.pth`三个文件，后续可以打包成`exe`可执行程序，便于在其他电脑上进行使用。

软件界面如下所示：

![image-20220817102327193](D:\Python\project\machine_hand_detection\pictures\image-20220817102327193.png)

<center>图2 软件操作界面</center>

软件支持两种数据选择方式，包含摄像头捕捉图像和电脑磁盘读取图像两种方式。选择/捕捉到对应的图像后将会在界面的图像框中进行显示。显示后便可以点击**开始预测按钮**，则软件开始加载`savemodel.pth`进行图像预测，并将模型预测的结果显示在软件的右上角方框中。

如果需要改变软件的窗口显示结果，则按照以下教程进行安装**pyqt5**，具体见：https://blog.csdn.net/qq_45041871/article/details/113775749。

如需打包程序成**exe**文件，只需在**terminal**跳转到**main.py**对应的根目录，然后执行命令**pyinstaller -F main.py**，等程序执行结束，将在在**main.py**的同级目录生成一个**dist**文件夹，文件夹中存在一个**main.exe**文件，然后将**savemodel.pth**存放在该**dist**文件夹下就可以执行**main.exe**文件。

# EMG识别

EMG手势识别也提供两份数据集，一份是公开的`SIA_delsys_16_movements_data`数据，另外一份是自己使用一个肌肉传感器采集的数据。两个数据集也存在不一样的数据处理方式，需要以一一对应，否则将会先问题。

## 数据处理

对于公开的`SIA_delsys_16_movements_data`数据，我们采用`./EMG/data/data.py`进行数据处理，处理后将会得到`image_feature.h5`和`time_feature.h5`文件。同样采用`./EMG/data/process_sku_data.py`进行数据处理，处理后将会得到`image_feature.h5`。这是模型的输入文件，必须严格控制其输入格式。

