import torch
import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights


def vgg11_model(x, weights):
    relu = nn.ReLU(inplace=True)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Feature Extractor
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    conv1.weight = nn.Parameter(weights["features.0.weight"])
    conv1.bias = nn.Parameter(weights["features.0.bias"])

    conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    conv2.weight = nn.Parameter(weights["features.3.weight"])
    conv2.bias = nn.Parameter(weights["features.3.bias"])

    conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    conv3.weight = nn.Parameter(weights["features.6.weight"])
    conv3.bias = nn.Parameter(weights["features.6.bias"])

    conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    conv4.weight = nn.Parameter(weights["features.8.weight"])
    conv4.bias = nn.Parameter(weights["features.8.bias"])

    conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    conv5.weight = nn.Parameter(weights["features.11.weight"])
    conv5.bias = nn.Parameter(weights["features.11.bias"])

    conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    conv6.weight = nn.Parameter(weights["features.13.weight"])
    conv6.bias = nn.Parameter(weights["features.13.bias"])

    conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    conv7.weight = nn.Parameter(weights["features.16.weight"])
    conv7.bias = nn.Parameter(weights["features.16.bias"])

    conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    conv8.weight = nn.Parameter(weights["features.18.weight"])
    conv8.bias = nn.Parameter(weights["features.18.bias"])

    avgpool = nn.AdaptiveAvgPool2d((7, 7))

    # Classifier
    fc1 = nn.Linear(512 * 7 * 7, 4096)
    fc1.weight = nn.Parameter(weights["classifier.0.weight"])
    fc1.bias = nn.Parameter(weights["classifier.0.bias"])

    fc2 = nn.Linear(4096, 4096)
    fc2.weight = nn.Parameter(weights["classifier.3.weight"])
    fc2.bias = nn.Parameter(weights["classifier.3.bias"])

    fc3 = nn.Linear(4096, 1000)
    fc3.weight = nn.Parameter(weights["classifier.6.weight"])
    fc3.bias = nn.Parameter(weights["classifier.6.bias"])

    # Forward pass
    x = pool(relu(conv1(x)))
    x = pool(relu(conv2(x)))
    x = relu(conv3(x))
    x = pool(relu(conv4(x)))
    x = relu(conv5(x))
    x = pool(relu(conv6(x)))
    x = relu(conv7(x))
    x = pool(relu(conv8(x)))

    x = avgpool(x)
    x = torch.flatten(x, 1)

    x = relu(fc1(x))
    x = relu(fc2(x))
    x = fc3(x)

    return x


def vgg11(x, weights=None):
    return vgg11_model(x, weights)
