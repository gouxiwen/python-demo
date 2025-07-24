def image_classification_example():
    import os
    import torch
    from torchvision.io import decode_image
    from torchvision.models import resnet18
    import torchvision
    from torchvision.transforms._presets import ImageClassification

    dog1 = decode_image("datasets01/test/83.jpg") #dog
    dog2 = decode_image("datasets01/test/2.jpg") #dog
    dog = {
        "real": "dog",
        "img": [dog1, dog2]
    }
    cat1 = decode_image("datasets01/test/84.jpg") #cat
    cat2 = decode_image("datasets01/test/6.jpg") #cat
    cat = {
        "real": "cat",
        "img": [cat1, cat2]
    }

    model = resnet18(num_classes=2)
    model_path = os.path.join("model/checkpoint.pth")
    model_data = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(model_data.get("model"))
    class_names = model_data.get("class_names", None)    
    print(f"{class_names=}")
    model.eval()

    preprocess =  ImageClassification(
        crop_size=224,
        resize_size=232,
    )

    # Step 3: Apply inference preprocessing transforms
    for dic in [dog, cat]:
        for img in dic["img"]:
            batch = preprocess(img).unsqueeze(0)
            # Step 4: Use the model and print the predicted category
            prediction = model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            print(f"{class_names[class_id]}: {100 * score:.1f}% -> real: {dic['real']}\n")


if __name__ == '__main__':
    image_classification_example()