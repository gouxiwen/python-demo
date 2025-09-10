# 探索transforms v2，它是一个新的图像变换API，代替了旧的transforms API。
# 文档：https://docs.pytorch.org/vision/0.22/transforms.html#transforms
# PyTorch 建议使用 torchvision.transforms.v2 变换而不是 torchvision.transforms 中的变换
from pathlib import Path
import torch
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.transforms import v2
from torchvision.io import decode_image

torch.manual_seed(1)

# If you're trying to run that on Colab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
from helpers import plot
img = decode_image(str(Path('./images') / 'astronaut.jpg'))
# print(f"{type(img) = }, {img.dtype = }, {img.shape = }")

# The basics
# 基础用法
# transform = v2.RandomCrop(size=(224, 224))
# out = transform(img)

# plot([img, out])


# I just want to do image classification
# If you just care about image classification, things are very simple. A basic classification pipeline may look like this:
# 如果你只关心图像分类，事情就很简单了。一个基本的分类图像转换流程可能如下所示：
# transforms = v2.Compose([
#     v2.RandomResizedCrop(size=(224, 224), antialias=True),
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# out = transforms(img)

# plot([img, out])

# 然后将transforms作为参数传递给数据集，如：ImageNet(..., transform=transforms)


# Detection, Segmentation, Videos
# transforms v2现在支持图像分类以外的任务：它们还可以转换边界框、分割/检测蒙版或视频
from torchvision import tv_tensors  # we'll describe this a bit later, bare with us

boxes = tv_tensors.BoundingBoxes(
    [
        [15, 10, 370, 510],
        [275, 340, 510, 510],
        [130, 345, 210, 425]
    ],
    format="XYXY", canvas_size=img.shape[-2:])

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomPhotometricDistort(p=1),
    v2.RandomHorizontalFlip(p=1),
])
out_img, out_boxes = transforms(img, boxes)
print(type(boxes), type(out_boxes))

plot([(img, boxes), (out_img, out_boxes)])

# 上面的示例侧重于对象检测。
# 但是，如果我们有用于对象分割或语义分割的掩码 (torchvision.tv_tensors.Mask)，
# 或者视频 (torchvision.tv_tensors.Video)，
# 我们也可以以完全相同的方式将它们传递给变换。