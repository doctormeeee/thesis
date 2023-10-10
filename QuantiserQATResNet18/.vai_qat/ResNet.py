# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ResNet(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResNet::input_0(ResNet::nndct_input_0)
        self.module_1 = py_nndct.nn.quant_input() #ResNet::ResNet/QuantStub[quant_stub]/3527(ResNet::nndct_quant_stub_1)
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Conv2d[conv1]/ret.5(ResNet::nndct_conv2d_2)
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/ReLU[relu]/3558(ResNet::nndct_relu_3)
        self.module_4 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #ResNet::ResNet/MaxPool2d[maxpool]/3573(ResNet::nndct_maxpool_4)
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/ret.9(ResNet::nndct_conv2d_5)
        self.module_6 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu1]/3602(ResNet::nndct_relu_6)
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]/ret.13(ResNet::nndct_conv2d_7)
        self.module_8 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/Add[skip_add]/ret.17(ResNet::nndct_elemwise_add_8)
        self.module_9 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu2]/3633(ResNet::nndct_relu_9)
        self.module_10 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]/ret.19(ResNet::nndct_conv2d_10)
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu1]/3661(ResNet::nndct_relu_11)
        self.module_12 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/ret.23(ResNet::nndct_conv2d_12)
        self.module_13 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/Add[skip_add]/ret.27(ResNet::nndct_elemwise_add_13)
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu2]/3692(ResNet::nndct_relu_14)
        self.module_15 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]/ret.29(ResNet::nndct_conv2d_15)
        self.module_16 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu1]/3720(ResNet::nndct_relu_16)
        self.module_17 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]/ret.33(ResNet::nndct_conv2d_17)
        self.module_18 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/ret.37(ResNet::nndct_conv2d_18)
        self.module_19 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/Add[skip_add]/ret.41(ResNet::nndct_elemwise_add_19)
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu2]/3778(ResNet::nndct_relu_20)
        self.module_21 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]/ret.43(ResNet::nndct_conv2d_21)
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu1]/3806(ResNet::nndct_relu_22)
        self.module_23 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]/ret.47(ResNet::nndct_conv2d_23)
        self.module_24 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/Add[skip_add]/ret.51(ResNet::nndct_elemwise_add_24)
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu2]/3837(ResNet::nndct_relu_25)
        self.module_26 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]/ret.53(ResNet::nndct_conv2d_26)
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu1]/3865(ResNet::nndct_relu_27)
        self.module_28 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]/ret.57(ResNet::nndct_conv2d_28)
        self.module_29 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/ret.61(ResNet::nndct_conv2d_29)
        self.module_30 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/Add[skip_add]/ret.65(ResNet::nndct_elemwise_add_30)
        self.module_31 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu2]/3923(ResNet::nndct_relu_31)
        self.module_32 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]/ret.67(ResNet::nndct_conv2d_32)
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu1]/3951(ResNet::nndct_relu_33)
        self.module_34 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]/ret.71(ResNet::nndct_conv2d_34)
        self.module_35 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/Add[skip_add]/ret.75(ResNet::nndct_elemwise_add_35)
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu2]/3982(ResNet::nndct_relu_36)
        self.module_37 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]/ret.77(ResNet::nndct_conv2d_37)
        self.module_38 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu1]/4010(ResNet::nndct_relu_38)
        self.module_39 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]/ret.81(ResNet::nndct_conv2d_39)
        self.module_40 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/ret.85(ResNet::nndct_conv2d_40)
        self.module_41 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/Add[skip_add]/ret.89(ResNet::nndct_elemwise_add_41)
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu2]/4068(ResNet::nndct_relu_42)
        self.module_43 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/ret.91(ResNet::nndct_conv2d_43)
        self.module_44 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu1]/4096(ResNet::nndct_relu_44)
        self.module_45 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]/ret.95(ResNet::nndct_conv2d_45)
        self.module_46 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/Add[skip_add]/ret.99(ResNet::nndct_elemwise_add_46)
        self.module_47 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu2]/4127(ResNet::nndct_relu_47)
        self.module_48 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[1, 1]) #ResNet::ResNet/AdaptiveAvgPool2d[avgpool]/4144(ResNet::nndct_adaptive_avg_pool2d_48)
        self.module_49 = py_nndct.nn.Module('nndct_flatten') #ResNet::ResNet/ret.101(ResNet::nndct_flatten_49)
        self.module_50 = py_nndct.nn.Linear(in_features=512, out_features=1000, bias=True) #ResNet::ResNet/Linear[fc]/ret.103(ResNet::nndct_dense_50)
        self.module_51 = py_nndct.nn.dequant_output() #ResNet::ResNet/DeQuantStub[dequant_stub]/4152(ResNet::nndct_dequant_stub_51)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_6(output_module_5)
        output_module_5 = self.module_7(output_module_5)
        output_module_5 = self.module_8(input=output_module_5, other=output_module_0, alpha=1)
        output_module_5 = self.module_9(output_module_5)
        output_module_10 = self.module_10(output_module_5)
        output_module_10 = self.module_11(output_module_10)
        output_module_10 = self.module_12(output_module_10)
        output_module_10 = self.module_13(input=output_module_10, other=output_module_5, alpha=1)
        output_module_10 = self.module_14(output_module_10)
        output_module_15 = self.module_15(output_module_10)
        output_module_15 = self.module_16(output_module_15)
        output_module_15 = self.module_17(output_module_15)
        output_module_18 = self.module_18(output_module_10)
        output_module_15 = self.module_19(input=output_module_15, other=output_module_18, alpha=1)
        output_module_15 = self.module_20(output_module_15)
        output_module_21 = self.module_21(output_module_15)
        output_module_21 = self.module_22(output_module_21)
        output_module_21 = self.module_23(output_module_21)
        output_module_21 = self.module_24(input=output_module_21, other=output_module_15, alpha=1)
        output_module_21 = self.module_25(output_module_21)
        output_module_26 = self.module_26(output_module_21)
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_29 = self.module_29(output_module_21)
        output_module_26 = self.module_30(input=output_module_26, other=output_module_29, alpha=1)
        output_module_26 = self.module_31(output_module_26)
        output_module_32 = self.module_32(output_module_26)
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_32 = self.module_35(input=output_module_32, other=output_module_26, alpha=1)
        output_module_32 = self.module_36(output_module_32)
        output_module_37 = self.module_37(output_module_32)
        output_module_37 = self.module_38(output_module_37)
        output_module_37 = self.module_39(output_module_37)
        output_module_40 = self.module_40(output_module_32)
        output_module_37 = self.module_41(input=output_module_37, other=output_module_40, alpha=1)
        output_module_37 = self.module_42(output_module_37)
        output_module_43 = self.module_43(output_module_37)
        output_module_43 = self.module_44(output_module_43)
        output_module_43 = self.module_45(output_module_43)
        output_module_43 = self.module_46(input=output_module_43, other=output_module_37, alpha=1)
        output_module_43 = self.module_47(output_module_43)
        output_module_43 = self.module_48(output_module_43)
        output_module_43 = self.module_49(input=output_module_43, start_dim=1, end_dim=-1)
        output_module_43 = self.module_50(output_module_43)
        output_module_43 = self.module_51(input=output_module_43)
        return output_module_43
