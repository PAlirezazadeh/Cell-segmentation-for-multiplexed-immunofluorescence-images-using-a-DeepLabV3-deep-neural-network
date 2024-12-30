#--------------------------------------------#
# This section of the code is used to view the network structure
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 2
    backbone        = 'mobilenet'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=16, pretrained=False).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2 is because profile does not count convolutions as two operations
    #   Some papers count convolutions as two operations: multiplication and addition. In this case, multiply by 2.
    #   Some papers only consider multiplication operations, ignoring addition. In this case, do not multiply by 2.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
