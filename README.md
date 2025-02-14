# VGG Inference Implementation


VGG means Visual Geometry Group. But before diving straight into VGG-11 and its specifications, it is of utmost important to have a glimpse of AlexNet (just the basic part). AlexNet is primarily capable of object-detection.It takes into acccount overfitting by using data augmentation and dropout. It replaces tanh activation function with ReLU by encapsulating its distinct features for over-pooling. VGG came into picture as it addresses the depth of CNNs.

# Details

reference -> Vgg implementation code
tests/test_vgg.py -> Unit test files to compare the custom built model and the pretrained model
tests/test_vgg_real_image.py -> Testing on the real data

# WorkFlow

- Clone the repo
- Install the required packages
- Move to tests directory and run `pytest test_vgg.py` or just `pytest`
- OR stay in the root directory(Vgg_Implementation) and run `pytest`. -> This will automatically detect the test files.


# Expected Results

- Pretrained Model Predicted Class Index: 208
- Custom Model Predicted Class Index: 208
    >>  Same class!
