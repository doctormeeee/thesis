# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ResNet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Conv2d[conv1]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/ReLU[relu]/2674
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #ResNet::ResNet/MaxPool2d[maxpool]/input.7
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/input.9
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/input.13
        self.module_6 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]/input.15
        self.module_7 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/input.17
        self.module_8 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/input.19
        self.module_9 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]/input.21
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/input.25
        self.module_11 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/input.27
        self.module_12 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/input.29
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/input.31
        self.module_14 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]/input.33
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/input.37
        self.module_16 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]/input.39
        self.module_17 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.41
        self.module_18 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/input.43
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/input.45
        self.module_20 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]/input.47
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/input.51
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]/input.53
        self.module_23 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/input.55
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/input.57
        self.module_25 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]/input.59
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/input.63
        self.module_27 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]/input.65
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.67
        self.module_29 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/input.69
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/input.71
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]/input.73
        self.module_32 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/input.77
        self.module_33 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]/input.79
        self.module_34 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/input.81
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/input.83
        self.module_36 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]/input.85
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/input.89
        self.module_38 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]/input.91
        self.module_39 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.93
        self.module_40 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/input.95
        self.module_41 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/input.97
        self.module_42 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/input.99
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/input.103
        self.module_44 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]/input.105
        self.module_45 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/input.107
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/input
        self.module_47 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #ResNet::ResNet/AdaptiveAvgPool2d[avgpool]/3211
        self.module_48 = py_nndct.nn.Module('nndct_flatten') #ResNet::ResNet/3214
        self.module_49 = py_nndct.nn.Linear(in_features=512, out_features=1000, bias=True) #ResNet::ResNet/Linear[fc]/3215

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_4 = self.module_6(output_module_4)
        output_module_4 = self.module_7(input=output_module_4, other=output_module_0, alpha=1)
        output_module_4 = self.module_8(output_module_4)
        output_module_9 = self.module_9(output_module_4)
        output_module_9 = self.module_10(output_module_9)
        output_module_9 = self.module_11(output_module_9)
        output_module_9 = self.module_12(input=output_module_9, other=output_module_4, alpha=1)
        output_module_9 = self.module_13(output_module_9)
        output_module_14 = self.module_14(output_module_9)
        output_module_14 = self.module_15(output_module_14)
        output_module_14 = self.module_16(output_module_14)
        output_module_17 = self.module_17(output_module_9)
        output_module_14 = self.module_18(input=output_module_14, other=output_module_17, alpha=1)
        output_module_14 = self.module_19(output_module_14)
        output_module_20 = self.module_20(output_module_14)
        output_module_20 = self.module_21(output_module_20)
        output_module_20 = self.module_22(output_module_20)
        output_module_20 = self.module_23(input=output_module_20, other=output_module_14, alpha=1)
        output_module_20 = self.module_24(output_module_20)
        output_module_25 = self.module_25(output_module_20)
        output_module_25 = self.module_26(output_module_25)
        output_module_25 = self.module_27(output_module_25)
        output_module_28 = self.module_28(output_module_20)
        output_module_25 = self.module_29(input=output_module_25, other=output_module_28, alpha=1)
        output_module_25 = self.module_30(output_module_25)
        output_module_31 = self.module_31(output_module_25)
        output_module_31 = self.module_32(output_module_31)
        output_module_31 = self.module_33(output_module_31)
        output_module_31 = self.module_34(input=output_module_31, other=output_module_25, alpha=1)
        output_module_31 = self.module_35(output_module_31)
        output_module_36 = self.module_36(output_module_31)
        output_module_36 = self.module_37(output_module_36)
        output_module_36 = self.module_38(output_module_36)
        output_module_39 = self.module_39(output_module_31)
        output_module_36 = self.module_40(input=output_module_36, other=output_module_39, alpha=1)
        output_module_36 = self.module_41(output_module_36)
        output_module_42 = self.module_42(output_module_36)
        output_module_42 = self.module_43(output_module_42)
        output_module_42 = self.module_44(output_module_42)
        output_module_42 = self.module_45(input=output_module_42, other=output_module_36, alpha=1)
        output_module_42 = self.module_46(output_module_42)
        output_module_42 = self.module_47(output_module_42)
        output_module_42 = self.module_48(input=output_module_42, start_dim=1, end_dim=-1)
        output_module_42 = self.module_49(output_module_42)
        return output_module_42
