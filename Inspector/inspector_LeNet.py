import torch
from torch import nn

from pytorch_nndct import Inspector

class MyLeNet5(nn.Module):

    def __init__(self):
        super(MyLeNet5, self).__init__()

        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2) 
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    # x = torch.rand([1, 3, 28, 28])
    # model = MyLeNet5()
    # y = model(x)

    # Specify a target name or fingerprint you want to deploy on
    target = "DPUCVDX8H_ISA1_F2W2_8PE"
    # Initialize inspector with target
    inspector = Inspector(target)

    # Start to inspect the float model
    # Note: visualization of inspection results relies on the dot engine.If you don't install dot successfully, set 'image_format = None' when inspecting.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyLeNet5()
    dummy_input = torch.randn(1, 3, 28, 28)
    inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect", image_format="png") 