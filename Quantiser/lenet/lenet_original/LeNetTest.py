import torch
from LeNet import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# import pytorch_nndct

data_transform = transforms.Compose([transforms.ToTensor()])


# train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# train_dataset_path = '~/Vitis-AI/Vitis-AI/Study_Workflow/Quantiser/MNIST/val' 
# test_dataset_path = '~/Vitis-AI/Vitis-AI/Study_Workflow/Quantiser/MNIST/val'

train_dataset_path = '/workspace/Study_Workflow/Quantiser/MNIST/val' 
test_dataset_path = '/workspace/Study_Workflow/Quantiser/MNIST/val'

test_dataset = datasets.ImageFolder(test_dataset_path, transforms.Compose([
            transforms.ToTensor(),
        ]))

train_dataloader = torch.utils.data.DataLoader(dataset=datasets.ImageFolder(train_dataset_path, transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=16, shuffle=True)


test_dataloader = torch.utils.data.DataLoader(dataset=datasets.ImageFolder(test_dataset_path, transforms.Compose([
            transforms.ToTensor(),
        ])), batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = MyLeNet5().to(device)

# model.load_state_dict(torch.load('~/Vitis-AI/Vitis-AI/Study_Workflow/Quantiser/model/lenet.pth')) 
# model.load_state_dict(torch.load('/home/neutronmgr/Vitis-AI/Vitis-AI/Study_Workflow/Quantiser/model/lenet.pth', map_location=torch.device('cpu')))
model.load_state_dict(torch.load('/workspace/Study_Workflow/Quantiser/model/lenet.pth', map_location=torch.device('cpu')))

# model.load_state_dict(torch.load('/workspace/Study_Workflow/Quantiser/quantize_result/MyLeNet5_int.pt'))#, map_location=torch.device('cpu')))

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

show = ToPILImage()

for i in range(10):
    X, y = test_dataset[979][0], test_dataset[979][1]
    # show(X).show()

    X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)

    with torch.no_grad():
        pred = model(X)
        print(pred)
        print(pred[0])

        predicted, actual = classes[torch.argmax(pred[0])], classes[y]

        print(f'predicted: "{predicted}", actual:"{actual}"')