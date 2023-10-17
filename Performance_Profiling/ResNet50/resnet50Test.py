import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
# from torchvision.transforms import ToPILImage
from torchvision.models.resnet import resnet50
from tqdm import tqdm
import time

def load_data(data_dir='', batch_size=''):

	valdir = data_dir + '/val'
	# valdir = data_dir
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	size = 224
	resize = 256
	
	dataset = datasets.ImageFolder(
		valdir,
		transforms.Compose([
			transforms.Resize(resize),
			transforms.CenterCrop(size),
			transforms.ToTensor(),
			normalize,
		]))
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, shuffle=False)
	return data_loader

# class AverageMeter(object):
# 	"""Computes and stores the average and current value"""

# 	def __init__(self, name, fmt=':f'):
# 		self.name = name
# 		self.fmt = fmt
# 		self.reset()

# 	def reset(self):
# 		self.val = 0
# 		self.avg = 0
# 		self.sum = 0
# 		self.count = 0

# 	def update(self, val, n=1):
# 		self.val = val
# 		self.sum += val * n
# 		self.count += n
# 		self.avg = self.sum / self.count

# 	def __str__(self):
# 		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
# 		return fmtstr.format(**self.__dict__)

# def accuracy(output, target, topk=(1,)):
# 	"""Computes the accuracy over the k top predictions
# 	for the specified values of k"""
# 	with torch.no_grad():
# 		maxk = max(topk)
# 		batch_size = target.size(0)

# 		_, pred = output.topk(maxk, 1, True, True)
# 		pred = pred.t()
# 		correct = pred.eq(target.view(1, -1).expand_as(pred))

# 		res = []
# 	for k in topk:
# 		correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
# 		res.append(correct_k.mul_(100.0 / batch_size))
# 	return res

def evaluate(model, val_loader, loss_fn):

	model.eval()
	model = model.to(device)
	# top1 = AverageMeter('Acc@1', ':6.2f')
	# top5 = AverageMeter('Acc@5', ':6.2f')
	total = 0
	Loss = 0
	for iteraction, (images, labels) in tqdm(
		enumerate(val_loader), total=len(val_loader)):
		# images.
		images = images.to(device)
		labels = labels.to(device)
		#pdb.set_trace()
		outputs = model(images)
		# print(f"Output tensor: {torch.argmax(outputs, axis=1)}")

		# loss = loss_fn(outputs, labels)
		# print(torch.argmax(outputs, axis=1))
		# print(outputs.size())
		# print(torch.max(outputs, axis=1)[1])
		# print(labels)
		# Loss += loss.item()
		# total += images.size(0)
		# acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
		# top1.update(acc1[0], images.size(0))
		# top5.update(acc5[0], images.size(0))
	# return top1.avg, top5.avg, Loss / total


if __name__ == "__main__":
	data_transform = transforms.Compose([transforms.ToTensor()])

	# device = "cuda" if torch.cuda.is_available() else 'cpu'
	
	device = 'cpu'

 
	model = resnet50().to(device)

	model.load_state_dict(torch.load("./model/resnet50.pth"))

	loss_fn = torch.nn.CrossEntropyLoss().to(device)

	# val_loader = load_data(data_dir='imagenet', batch_size=10)
	val_loader = load_data(data_dir='imagenet', batch_size=8)

	start_time = time.time()
 

	acc1_gen, acc5_gen, loss_gen = evaluate(model, val_loader, loss_fn)
 
	end_time = time.time()
 
	execution_time = end_time - start_time
 
	print(f'Execution time: {execution_time}')

	# logging accuracy
	print('loss: %g' % (loss_gen))
	print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

# show = ToPILImage()

# for i in range(20):
# 	X, y = test_dataset[i][0], test_dataset[i][1]
# 	show(X).show()

# 	X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)

# 	with torch.no_grad():
# 		pred = model(X)

# 		predicted, actual = classes[torch.argmax(pred[0])], classes[y]

# 		print(f'predicted: "{predicted}", actual:"{actual}"')