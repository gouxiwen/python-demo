# 使用yolo11进行预测，包括以下五种任务（训练模型的数据集）：
# Detection (COCO)
# Segmentation (COCO)
# Classification (ImageNet)
# Pose (COCO)
# Oriented Bounding Boxes (DOTAv1)
# 使用文档：https://docs.ultralytics.com/zh/modes/predict/#introduction
# 模型下载：https://github.com/ultralytics/ultralytics?tab=readme-ov-file
from ultralytics import YOLO
import os

# eg1:Detection
# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

def detectImage():
    # Run batched inference on a list of images
    img1 = os.path.join('../images/detection1.jpg')
    img2 = os.path.join('../images/detection2.jpg')
    results = model([img1, img2])  # return a list of Results objects

    # Process results list
    for index, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        # result.save(filename=f"result{index}.jpg")  # save to disk

def detectVideo():
    # Define path to video file
    source = os.path.join('../video/test.mp4')
    # Run inference on the source
    results = model(source, stream=True)  # generator of Results objects
    result = next(results)
    result.show()
    # result.save("result.jpg")

if __name__ == "__main__":
    # detectImage()
    detectVideo()