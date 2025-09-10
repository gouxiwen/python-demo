# 探索Optical Flow（光流）
# 使用 RAFT 模型预测运动
# 光流是预测两幅图像（通常是视频的两个连续帧）之间运动的任务。
# 光流模型以两幅图像作为输入，并预测一个光流：该光流指示第一幅图像中每个像素的位移，并将其映射到第二幅图像中对应的像素。
# 光流是 (2, H, W) 维张量，其中第一个维度对应于预测的水平和垂直位移。
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = "tight"

# 封装多图像绘制函数
def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()

# 下载视频
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
_ = urlretrieve(video_url, video_path)

# 读取视频帧
from torchvision.io import read_video
# read_video() 返回视频帧、音频帧以及与视频相关的元数据。在本例中，我们只需要视频帧。
frames, _, _ = read_video(str(video_path), output_format="TCHW")

# 批量预测，这里一次预测两张图像
img1_batch = torch.stack([frames[100], frames[150]])
img2_batch = torch.stack([frames[101], frames[151]])

# 如果想一次预测多张
# img1_batch = torch.stack([frames[100], frames[150], frames[200], frames[250]])
# img2_batch = torch.stack([frames[101], frames[151], frames[201], frames[251]])

# plot(img1_batch)

# 数据转换
from torchvision.models.optical_flow import Raft_Large_Weights,Raft_Small_Weights

weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()

# RAFT 模型接受 RGB 图像。我们首先从 read_video() 获取帧，并调整其大小以确保其尺寸可以被 8 整除。
# 注意，我们明确使用了 antialias=False，因为这些模型就是以此方式训练的。
# 然后，我们使用捆绑在权重中的变换来预处理输入，并将其值重新缩放到所需的 [-1, 1] 区间。
def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)


img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

# print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

# 预测
from torchvision.models.optical_flow import raft_small

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

# 可视化预测流
from torchvision.utils import flow_to_image

flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)