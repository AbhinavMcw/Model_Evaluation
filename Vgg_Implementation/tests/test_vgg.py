import torch
import numpy as np
from torchvision.models import vgg11, VGG11_Weights
from scipy.stats import pearsonr
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reference.vgg import vgg11 as functional_vgg11


def test_vgg11_pcc():
  
    pretrained_vgg11 = vgg11(weights=VGG11_Weights.DEFAULT)
    pretrained_vgg11.eval()

    
    weights = pretrained_vgg11.state_dict()

    x = torch.randn(1, 3, 224, 224) 

   
    with torch.no_grad():
        output_pretrained = pretrained_vgg11(x)
        output_functional = functional_vgg11(x, weights)

    # Convert tensors to numpy
    output_pretrained_np = output_pretrained.numpy().flatten()
    output_functional_np = output_functional.numpy().flatten()

   
    pcc, _ = pearsonr(output_pretrained_np, output_functional_np)
    
    print(pcc)

   
    assert pcc > 0.99, f"PCC score is too low: {pcc}"
