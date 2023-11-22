# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ToyModel(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ToyModel::input_0(ToyModel::nndct_input_0)
        self.module_1 = py_nndct.nn.Interpolate() #ToyModel::ToyModel/Upsample[upsample]/ret.3(ToyModel::nndct_resize_1)
        self.module_2 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ToyModel::ToyModel/Conv2d[conv]/ret.5(ToyModel::nndct_conv2d_2)
        self.module_3 = py_nndct.nn.ReLU(inplace=False) #ToyModel::ToyModel/ReLU[relu]/ret.7(ToyModel::nndct_relu_3)
        self.module_4 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[2, 2]) #ToyModel::ToyModel/AdaptiveAvgPool2d[adaptive_avg_pool2d]/230(ToyModel::nndct_adaptive_avg_pool2d_4)
        self.module_5 = py_nndct.nn.Module('nndct_permute') #ToyModel::ToyModel/ret.9(ToyModel::nndct_permute_5)
        self.module_6 = py_nndct.nn.Module('nndct_shape') #ToyModel::ToyModel/240(ToyModel::nndct_shape_6)
        self.module_7 = py_nndct.nn.Module('nndct_reshape') #ToyModel::ToyModel/ret(ToyModel::nndct_reshape_7)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(dims=[0,2,3,1], input=output_module_0)
        output_module_6 = self.module_6(input=output_module_0, dim=0)
        output_module_7 = self.module_7(input=output_module_0, shape=[output_module_6,-1])
        return output_module_7
