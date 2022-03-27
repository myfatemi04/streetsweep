from torchvision import transforms
from PIL import Image
import time
import cv2
import torch
import pandas as pd
import numpy as np

from imagechopper import chop_image

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()

# sample execution (requires torchvision)
input_image = Image.open('./litter0.jpg')
input_image = transforms.ToTensor()(input_image)

# input_tensor = preprocess(input_image)
bounding_boxes, input_batch = chop_image(input_image, 4, 4)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_batch = torch.stack([preprocess(i) for i in input_batch])
# create a mini-batch as expected by the model
# input_batch = input_tensor.unsqueeze(0)

# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

with torch.no_grad():
    output = model(input_batch)

for i in range(len(output)):
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[i], dim=0)

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
