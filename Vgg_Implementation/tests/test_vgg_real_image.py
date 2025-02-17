# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from transformers import ResNetForImageClassification
# from datasets import load_dataset
# import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from reference.vgg import vgg11 as custom_vgg11  # Import our VGG11

# # Load the pretrained VGG11 model
# pretrained_vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
# pretrained_vgg11.eval()

# # Extract pretrained weights to use in our custom model
# weights = pretrained_vgg11.state_dict()

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize image to match VGG input
#     transforms.ToTensor(),          # Convert image to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
# ])

# # Load an image from Hugging Face dataset
# dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
# image = dataset["test"]["image"][0]
# image = transform(image).unsqueeze(0)  # Add batch dimension

# # Run inference on both models
# with torch.no_grad():
#     output_pretrained = pretrained_vgg11(image)  # Pretrained model output
#     output_custom = custom_vgg11(image, weights)  # Our custom VGG11 output

# # Convert output logits to class index
# predicted_class_pretrained = output_pretrained.argmax(dim=1).item()
# predicted_class_custom = output_custom.argmax(dim=1).item()

# # Load class labels from ResNet model to get human-readable labels
# transformer_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# print("Pretrained VGG11 Predicted Class:", transformer_model.config.id2label[predicted_class_pretrained])
# print("Custom VGG11 Predicted Class:", transformer_model.config.id2label[predicted_class_custom])

# # Check if both models give the same class prediction
# if predicted_class_pretrained == predicted_class_custom:
#     print("Both models predicted the same class!")
# else:
#     print("Mismatch in predicted classes!")

import torch
import pytest
from torchvision import models, transforms
from datasets import load_dataset
from transformers import ResNetForImageClassification
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reference.vgg import vgg11 as custom_vgg11

def test_vgg11_inference():
    
    transformer_model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50"
    )
    
   
    pretrained_vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    pretrained_vgg11.eval()
    
   
    weights = pretrained_vgg11.state_dict()
    
   
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]
    
  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    data = transform(image).unsqueeze(0)  
    
    
    with torch.no_grad():
        output_pretrained = pretrained_vgg11(data)
        output_custom = custom_vgg11(data, weights)
    
   
    predicted_class_pretrained = output_pretrained.argmax(dim=1).item()
    predicted_class_custom = output_custom.argmax(dim=1).item()
    
   
    pretrained_label = transformer_model.config.id2label.get(predicted_class_pretrained, "Unknown")
    custom_label = transformer_model.config.id2label.get(predicted_class_custom, "Unknown")
    
   
    print(f"Pretrained Model Predicted Class: {pretrained_label} ({predicted_class_pretrained})")
    print(f"Custom Model Predicted Class: {custom_label} ({predicted_class_custom})")
    
    
    assert predicted_class_pretrained == predicted_class_custom, (
        f"Mismatch in predicted classes! Pretrained: {pretrained_label}, Custom: {custom_label}"
    )
