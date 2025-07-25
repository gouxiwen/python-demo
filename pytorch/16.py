# 探索pytorch扩展库torchvision（视觉）和torchtext（文本）
# TorchVision 是 PyTorch 生态系统中用于计算机视觉任务的扩展库，提供预训练模型、常用数据集和图像处理工具，支持图像分类、像素语义分割、物体检测、实例分割、人物关键点检测、视频分类和光流（预测运动）。 ‌
# 核心功能
# TorchVision 包含四个核心模块：
# ‌数据集（datasets）‌：支持 CIFAR-10 、 MNIST 、 ImageNet 等常用数据集，可直接加载或下载 ‌
# ‌模型（models）‌：包含 ResNet 、 AlexNet 、 VGG 等预训练模型，支持迁移学习 ‌
# ‌转换（transforms）‌：提供数据增强和预处理功能，如归一化、裁剪等 ‌
# ‌工具（utils）‌：包含视频处理、模型可视化等辅助功能 ‌

# 测试数据地址：https://github.com/pytorch/vision/tree/main/test

import torch
import numpy as np
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
# from torchtext import models

# 列出所有模型
# all_models = list_models()
# print(all_models)

# 列出分类模型
# classification_models = list_models(module=models)
# print(classification_models)

# Fetch weights
# weights = get_weight("MobileNet_V3_Large_Weights.DEFAULT")
# assert weights == MobileNet_V3_Large_Weights.DEFAULT
# Here is an example of how to use the pre-trained models:

plt.rcParams["savefig.bbox"] = 'tight'

# 同时绘制多个图像
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
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 图像分类模型Image Classification
def image_classification_example():
    from torchvision.io import decode_image
    from torchvision.models import resnet50, ResNet50_Weights ,EfficientNet
    # resnet残差网络，有不同层数的版本，18, 34, 50, 101, 152，层数越多，模型越深，参数量越大，效果越好，但计算量也越大
    # 残差网络是v0.13版本之前torchvision中内置的最好的模型，不过v0.13版本之后加入了EfficientNet效果更好

    # img = decode_image("mini-hymenoptera_data/val/bees/54736755_c057723f64.jpg")
    # img = decode_image("images/banana.jpg")
    img = decode_image("train-demo/classification/datasets01/test/84.jpg")

    # print(img.shape) #torch.Size([3, 396, 500]) channel, height, width
    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    # print(weights.meta["categories"])
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction: torch.Tensor = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 语义分割模型Semantic Segmentation（有三大类）
# DeepLabV3
# FCN
# LRASPP
# 列出所有语义分割模型
# segmentation_models = list_models(module=models.segmentation)
# print(segmentation_models)

def semantic_segmentation_example():
    from torchvision.io.image import decode_image
    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    from torchvision.transforms.functional import to_pil_image

    # img = decode_image("images/dog.jpg")
    # img = decode_image("images/cat.jpg")
    img = decode_image("images/person.jpg")

    # # Step 1: Initialize model with the best available weights
    weights = FCN_ResNet50_Weights.DEFAULT
    print(weights.meta["categories"])
    model = fcn_resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    # # mask = normalized_masks[0, class_to_idx["dog"]]
    # # mask = normalized_masks[0, class_to_idx["cat"]]
    mask = normalized_masks[0, class_to_idx["person"]]
    to_pil_image(mask).show()
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 物体检测模型Object Detection
# 物体检测模型基于分类模型进行初始化，生成Tensor[C, H, W]列表
# Faster R-CNN
# FCOS
# RetinaNet
# SSD
# SSDlite

def object_detection_example():
    from torchvision.io.image import decode_image
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import to_pil_image

    # img = decode_image("images/dog.jpg")
    img = decode_image("images/detection1.jpg")
    # img = decode_image("images/detection2.jpg")

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # print(weights.meta["categories"])
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4, font="c:/WINDOWS/Fonts/SIMSUN.TTC", font_size=30)
    im = to_pil_image(box.detach())
    im.show()
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 实例分割模型Instance Segmentation
# 1. 语义分割和实例分割的目标区别其实这两者区别是很简单的，语义分割和实例分割都是用于将图像分割成多个不同的部分，但他们的区别在于分割的目标不同：
# 语义分割：将图像中的每个像素分配给其对应的语义类别，其主要针对于像素，或者说它是像素级别的图像分割方法。（同一类别的物体会被分到同一个区域）
# 实例分割：将图像中的每个物体分割成独立的实例，简单地说，就是将图像中的每个个体对象分开。例如，在一张人群照片中分割出每个人的轮廓。（同一类别不同的个体会被分到不同的区域）
# 也就是说实例分割在语义分割的基础上增加了对同一类别不同个体的区分。
# Mask R-CNN
def instance_segmentation_example():
    from torchvision.io.image import decode_image
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.utils import draw_segmentation_masks

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
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 关键点检测模型Keypoint Detection
# 关键点检测技术的应用
# 人机交互：通过关键点检测技术，计算机可以识别出人的手势和动作，从而实现与人的自然交互。例如，在智能家居系统中，我们可以通过手势控制灯光、空调等设备。
# 运动分析：在体育竞技、康复训练等领域，关键点检测技术可以用于分析运动员或康复者的运动轨迹和姿态变化，从而为训练和康复提供科学依据。
# 智能监控：在安防领域，关键点检测技术可以用于识别异常行为，如跌倒、打架等，从而实现智能监控和预警。
def keypoint_detection_example():
    import torch
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
    from torchvision.io import decode_image
    from torchvision.utils import draw_keypoints

    # person_int = decode_image('images/person.jpg')
    person_int = decode_image('images/person1.jpg')

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    print(weights.meta["categories"]) #['no person', 'person']
    transforms = weights.transforms()

    person_float = transforms(person_int)

    model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    outputs = model([person_float])

    kpts = outputs[0]['keypoints']
    scores = outputs[0]['scores']

    detect_threshold = 0.75
    idx = torch.where(scores > detect_threshold)
    keypoints = kpts[idx]
    # res = draw_keypoints(person_int, keypoints, colors="blue", radius=3)
    # show(res)

    # The coco keypoints for a person are ordered and represent the following list.
    # 人体关键点坐标顺序列表
    coco_keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    # 根据以上坐标顺序连接人体关键点形成骨架
    connect_skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    # res = draw_keypoints(person_int, keypoints, connectivity=connect_skeleton, colors="blue", radius=4, width=3)
    # show(res)
    # 加入有一些关键点没有预测出来，如下
    keypoints = torch.tensor(
        [[[208.0176, 214.2409, 1.0000],
          [000.0000, 000.0000, 0.0000],
          [197.8246, 210.6392, 1.0000],
          [000.0000, 000.0000, 0.0000],
          [178.6378, 217.8425, 1.0000],
          [221.2086, 253.8591, 1.0000],
          [160.6502, 269.4662, 1.0000],
          [243.9929, 304.2822, 1.0000],
          [138.4654, 328.8935, 1.0000],
          [277.5698, 340.8990, 1.0000],
          [153.4551, 374.5145, 1.0000],
          [000.0000, 000.0000, 0.0000],
          [226.0053, 370.3125, 1.0000],
          [221.8081, 455.5516, 1.0000],
          [273.9723, 448.9486, 1.0000],
          [193.6275, 546.1933, 1.0000],
          [273.3727, 545.5930, 1.0000]]]
    )

    # 不可见的关键点
    coordinates, visibility = keypoints.split([2, 1], dim=-1)
    visibility = visibility.bool()

    # 根据可见性绘制人体关键点
    res = draw_keypoints(
        person_int, coordinates, visibility=visibility, connectivity=connect_skeleton, colors="blue", radius=4, width=3
    )
    show(res)
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 视频分类模型Video Classification
def video_classification_example():
    from torchvision.io.video import read_video
    from torchvision.models.video import r3d_18, R3D_18_Weights

    vid, _, _ = read_video("images/v_SoccerJuggling_g23_c01.avi", output_format="TCHW")
    vid = vid[:32]  # optionally shorten duration

    # Step 1: Initialize model with the best available weights
    weights = R3D_18_Weights.DEFAULT
    print(weights.meta["categories"])
    model = r3d_18(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(vid).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    label = prediction.argmax().item()
    score = prediction[label].item()
    category_name = weights.meta["categories"][label]
    print(f"{category_name}: {100 * score}%")

image_classification_example()
# semantic_segmentation_example()
# object_detection_example()
# keypoint_detection_example()
# video_classification_example()
