# 探索pytorch扩展库torchvision（视觉）和torchtext（文本）
# TorchVision 是 PyTorch 生态系统中用于计算机视觉任务的扩展库，提供预训练模型、常用数据集和图像处理工具，支持图像分类、像素语义分割、物体检测、实例分割、人物关键点检测、视频分类和光流。 ‌
# 核心功能
# TorchVision 包含四个核心模块：
# ‌数据集（datasets）‌：支持 CIFAR-10 、 MNIST 、 ImageNet 等常用数据集，可直接加载或下载 ‌
# ‌模型（models）‌：包含 ResNet 、 AlexNet 、 VGG 等预训练模型，支持迁移学习 ‌
# ‌转换（transforms）‌：提供数据增强和预处理功能，如归一化、裁剪等 ‌
# ‌工具（utils）‌：包含视频处理、模型可视化等辅助功能 ‌
from torchvision import models
from torchvision.models import list_models, get_weight, MobileNet_V3_Large_Weights
# from torchtext import models

# 所有模型
# all_models = list_models()
# print(all_models)

# 分类模型
# classification_models = list_models(module=models)
# print(classification_models)

# Fetch weights
# weights = get_weight("MobileNet_V3_Large_Weights.DEFAULT")
# assert weights == MobileNet_V3_Large_Weights.DEFAULT

# Here is an example of how to use the pre-trained image classification models:
# from torchvision.io import decode_image
# from torchvision.models import resnet50, ResNet50_Weights ,EfficientNet
# # resnet残差网络，有不同层数的版本，18, 34, 50, 101, 152，层数越多，模型越深，参数量越大，效果越好，但计算量也越大
# # 残差网络是v0.13版本之前torchvision中内置的最好的模型，不过v0.13版本之后加入了EfficientNet效果更好

# # img = decode_image("mini-hymenoptera_data/val/bees/54736755_c057723f64.jpg")
# img = decode_image("images/banana.jpg")

# # print(img.shape) #torch.Size([3, 396, 500]) channel, height, width
# # Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# # print(weights.meta["categories"])
# model = resnet50(weights=weights)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")

# 语义分割模型Semantic Segmentation（有三大类）
# DeepLabV3
# FCN
# LRASPP
# 列出所有语义分割模型
# segmentation_models = list_models(module=models.segmentation)
# print(segmentation_models)

# Here is an example of how to use the pre-trained semantic segmentation models:
# from torchvision.io.image import decode_image
# from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
# from torchvision.transforms.functional import to_pil_image

# # img = decode_image("images/dog.jpg")
# # img = decode_image("images/cat.jpg")
# img = decode_image("images/person.jpg")

# # # Step 1: Initialize model with the best available weights
# weights = FCN_ResNet50_Weights.DEFAULT
# print(weights.meta["categories"])
# model = fcn_resnet50(weights=weights)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)

# # Step 4: Use the model and visualize the prediction
# prediction = model(batch)["out"]
# normalized_masks = prediction.softmax(dim=1)
# class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
# # # mask = normalized_masks[0, class_to_idx["dog"]]
# # # mask = normalized_masks[0, class_to_idx["cat"]]
# mask = normalized_masks[0, class_to_idx["person"]]
# to_pil_image(mask).show()

# 物体检测模型Object Detection
# 物体检测模型基于分类模型进行初始化，生成Tensor[C, H, W]列表
# Faster R-CNN
# FCOS
# RetinaNet
# SSD
# SSDlite

# from torchvision.io.image import decode_image
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# from torchvision.utils import draw_bounding_boxes
# from torchvision.transforms.functional import to_pil_image

# # img = decode_image("images/dog.jpg")
# img = decode_image("images/detection2.jpg")

# # Step 1: Initialize model with the best available weights
# weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# # print(weights.meta["categories"])
# model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()

# # Step 3: Apply inference preprocessing transforms
# batch = [preprocess(img)]

# # Step 4: Use the model and visualize the prediction
# prediction = model(batch)[0]
# labels = [weights.meta["categories"][i] for i in prediction["labels"]]
# box = draw_bounding_boxes(img, boxes=prediction["boxes"],
#                           labels=labels,
#                           colors="red",
#                           width=4, font="c:\WINDOWS\Fonts\SIMSUN.TTC", font_size=30)
# im = to_pil_image(box.detach())
# im.show()

# 实例分割模型Instance Segmentation
# 1. 语义分割和实例分割的目标区别其实这两者区别是很简单的，语义分割和实例分割都是用于将图像分割成多个不同的部分，但他们的区别在于分割的目标不同：
# 语义分割：将图像中的每个像素分配给其对应的语义类别，其主要针对于像素，或者说它是像素级别的图像分割方法。（同一类别的物体会被分到同一个区域）
# 实例分割：将图像中的每个物体分割成独立的实例，简单地说，就是将图像中的每个个体对象分开。例如，在一张人群照片中分割出每个人的轮廓。（同一类别不同的个体会被分到不同的区域）
# 也就是说实例分割在语义分割的基础上增加了对同一类别不同个体的区分。
# Mask R-CNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io.image import decode_image
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_segmentation_masks

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

dog1_int = decode_image("images/dogs.jpg")
cat1_int = decode_image("images/cats.jpg")
dog_list = [dog1_int,cat1_int]
images = [transforms(d) for d in dog_list]

model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()
output = model(images)

proba_threshold = 0.5 
score_threshold = .75
# 根据预测的分数和概率阈值觉得是否绘制mask
boolean_masks = [
    out['masks'][out['scores'] > score_threshold] > proba_threshold
    for out in output
]

dogs_with_masks = [
    #  There's an extra dimension (1) to the masks. We need to remove it
    draw_segmentation_masks(img, mask.squeeze(1), alpha=0.9)
    for img, mask in zip(dog_list, boolean_masks)
]
show(dogs_with_masks)