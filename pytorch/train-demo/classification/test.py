def image_classification_example():
    import os
    import torch
    from torchvision.io import decode_image
    from torchvision.models import resnet18
    import torchvision
    from torchvision.transforms._presets import ImageClassification

    img = decode_image("datasets01/test/83.jpg")
    # img = decode_image("datasets01/test/2.jpg")

    model = resnet18(num_classes=2)
    model_path = os.path.join("checkpoint.pth")
    model_data = torch.load(model_path,weights_only=False)
    model.load_state_dict(model_data.get("model"))
    class_names = model_data.get("class_names", None)    
    model.eval()

    preprocess =  ImageClassification(
        crop_size=224,
        resize_size=232,
    )

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    print(f"prediction = {prediction}")
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    print(f"{class_names[class_id]}: {100 * score:.1f}%")


if __name__ == '__main__':
    image_classification_example()