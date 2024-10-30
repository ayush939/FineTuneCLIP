import torch
import clip
import json
import os
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

val_dataset = torchvision.datasets.ImageNet(root="/fast_storage/mittal/", split="val", transform=preprocess)

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=B, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)



filename = "./../imagenet-class_names.json"
with open(filename) as f:
    imagenet_classes = json.load(f)
imagenet_classes = ["a photo of {}".format(cl) for cl in imagenet_classes]
encoded_classes = clip.tokenize(imagenet_classes).to(device)



with torch.no_grad():
    for imgs, labels in val_loader:
    #image_features = model.encode_image(image)
    #text_features = model.encode_text(text)
    
        logits_per_image, logits_per_text = model(imgs.to(device), encoded_classes)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        break

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
