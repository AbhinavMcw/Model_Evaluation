import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg11, VGG11_Weights
from PIL import Image
import requests
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reference.vgg import vgg11 as custom_vgg11  


pretrained_vgg11 = vgg11(weights=VGG11_Weights.DEFAULT)
pretrained_vgg11.eval()

weights = pretrained_vgg11.state_dict()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


image_url = "https://cdn.shopify.com/s/files/1/0086/0795/7054/files/Golden-Retriever.jpg?v=1645179525"
image = Image.open(requests.get(image_url, stream=True).raw)
image = transform(image).unsqueeze(0)  


with torch.no_grad():
    output_pretrained = pretrained_vgg11(image)  
    output_custom = custom_vgg11(image, weights) 


predicted_class_pretrained = output_pretrained.argmax(dim=1).item()
predicted_class_custom = output_custom.argmax(dim=1).item()


print("Pretrained VGG11 Output:", output_pretrained)
print("Custom VGG11 Output:", output_custom)
print("Pretrained Model Predicted Class Index:", predicted_class_pretrained)
print("Custom Model Predicted Class Index:", predicted_class_custom)


if predicted_class_pretrained == predicted_class_custom:
    print(" Same class!")
else:
    print("Mismatch!")
