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

def detectImage():
    # Load a model
    model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
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
     # Load a model
    model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
    # Define path to video file
    source = os.path.join('../video/test.mp4')
    # Run inference on the source
    results = model(source, stream=True)  # generator of Results objects
    result = next(results)
    result.show()
    # result.save("result.jpg")

# eg2: Segmentation
# YOLO11 中的对象检测和实例分割有什么区别？
# 物体检测通过绘制物体周围的边界框来识别和定位图像中的物体，而实例分割不仅能识别边界框，还能勾勒出每个物体的精确形状
def segmentImage():
    # Load a model
    model = YOLO("yolo11n-seg.pt")  # load an official model
    # model = YOLO("path/to/best.pt")  # load a custom model
    # Predict with the model
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # Access the results
    for result in results:
        xy = result.masks.xy  # mask in polygon format
        xyn = result.masks.xyn  # normalized
        masks = result.masks.data  # mask in matrix format (num_objects x H x W)
        result.show()
def segmentVideo():
    # Load a model
    model = YOLO("yolo11n-seg.pt")  # load an official model
    # model = YOLO("path/to/best.pt")  # load a custom model
    # Predict with the model
    results = model(os.path.join('../video/test.mp4'), stream=True) 
    # Access the results
    result = next(results)
    result.show()

# eg3: Classification
def classifyImage():
    # Load a model
    model = YOLO("yolo11n-cls.pt") 
    # Predict with the model
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    results = model(os.path.join('../images/dog.jpg')) # predict on an image
    # Access the results
    for result in results:
        print(f'result.probs.top1={result.names[result.probs.top1]}')
        print(f'result.probs.top5={[result.names[r] for r in result.probs.top5]}')
        # result.show()

# eg4: Pose 通过关键点检测来识别图像中的人体姿势
def poseImage():
    model = YOLO("yolo11n-pose.pt")
    results = model(os.path.join('../images/person1.jpg'))
    for result in results:
        print(result.keypoints.data)
        result.show()

# eg5: Oriented Bounding Boxes 
# 定向对象检测通过引入一个额外的角度来更准确地定位图像中的对象，从而比标准对象检测更进一步。
# 当目标以各种角度出现时，定向边界框特别有用，例如在航空图像中，传统的轴对齐边界框可能包含不必要的背景。
def obbImage():
    model = YOLO("yolo11n-obb.pt")
    results = model('https://ultralytics.com/images/boats.jpg')
    for result in results:
        print(result.obb.data)
        result.show()

if __name__ == "__main__":
    # detectImage()
    # detectVideo()
    # segmentImage()
    # segmentVideo()
    # classifyImage()
    # poseImage()
    obbImage()