# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class MyLeNet5(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(MyLeNet5, self).__init__()
        self.module_0 = py_nndct.nn.Input() #MyLeNet5::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #MyLeNet5::MyLeNet5/Conv2d[c1]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #MyLeNet5::MyLeNet5/ReLU[relu]/243
        self.module_3 = py_nndct.nn.AvgPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False, count_include_pad=True) #MyLeNet5::MyLeNet5/AvgPool2d[s2]/input.5
        self.module_4 = py_nndct.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MyLeNet5::MyLeNet5/Conv2d[c3]/input.7
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #MyLeNet5::MyLeNet5/ReLU[relu]/276
        self.module_6 = py_nndct.nn.AvgPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], ceil_mode=False, count_include_pad=True) #MyLeNet5::MyLeNet5/AvgPool2d[s4]/input
        self.module_7 = py_nndct.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=[5, 5], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MyLeNet5::MyLeNet5/Conv2d[c5]/308
        self.module_8 = py_nndct.nn.Module('nndct_flatten') #MyLeNet5::MyLeNet5/Flatten[flatten]/311
        self.module_9 = py_nndct.nn.Linear(in_features=120, out_features=84, bias=True) #MyLeNet5::MyLeNet5/Linear[f6]/312
        self.module_10 = py_nndct.nn.Linear(in_features=84, out_features=10, bias=True) #MyLeNet5::MyLeNet5/Linear[output]/313

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(input=output_module_0, start_dim=1, end_dim=-1)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(output_module_0)
        return output_module_0
