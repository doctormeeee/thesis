import torch
# Import inspector API
# 
# Note:
# You can ignore warning message related with XIR. 
# The inspector relies on 'vai_utf' package. In conda env vitis-ai-pytorch in Vitis-AI docker, vai_utf is ready. But if vai_q_pytorch is installed by source code, it needs to install vai_utf in advance.
from pytorch_nndct import Inspector

# Define a toy neural network
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = torch.nn.ReLU()
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size=2)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.adaptive_avg_pool2d(x)
        x = x.permute(0, 2, 3, 1) 
        x = x.reshape(x.size(0), -1)
        return x


if __name__ == "__main__":
    # Specify a target name or fingerprint you want to deploy on
    # target = "DPUCAHX8L_ISA0_SP"
    target = "DPUCVDX8H_ISA1_F2W2_8PE"
    # Initialize inspector with target
    inspector = Inspector(target)

    # Start to inspect the float model
    # Note: visualization of inspection results relies on the dot engine.If you don't install dot successfully, set 'image_format = None' when inspecting.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToyModel()
    dummy_input = torch.randn(1, 128, 3, 3)
    inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect", image_format="png") 