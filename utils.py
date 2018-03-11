import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import shutil
import os
from time import time

class Trainer:
	def __init__(self,**kwargs):
		self.model = kwargs['model']
		self.optimizer = kwargs['optimizer']
		self.criterion = kwargs['criterion']
		self.start_epoch = kwargs['start_epoch']
		self.epochs = kwargs['epochs']
		self.cuda = kwargs['cuda']
		self.log_interval = kwargs['log_interval']
		self.train_loader = kwargs['train_loader']
		self.test_loader = kwargs['test_loader']
		self.root = kwargs['root']
		if not os.path.exists(self.root):
			os.mkdir(self.root)

	def train(self,e):
		self.model.train()
		correct = 0
		train_size =0
		for batch_idx,(data,label) in enumerate(self.train_loader):
			if self.cuda:
				data,label = data.cuda(),label.cuda()
			data,label = Variable(data),Variable(label)
			self.optimizer.zero_grad()
			output = self.model(data,label)
			loss = self.criterion(output,label)
			loss.backward()
			self.optimizer.step()
			pred = output.data.max(1,keepdim=True)[1]
			correct += pred.eq(label.data.view_as(pred)).cpu().sum()
			train_size += len(data)
			if (batch_idx + 1) % self.log_interval == 0:
				print("Epoch: {} [{}/{} ({:.2f}%)]\t Loss: {:.6f} Acc: {:.6f}".format(
					e,
					(batch_idx + 1) * len(data),
					len(self.train_loader.dataset),
					100. * (batch_idx + 1) / len(self.train_loader),
					loss.data[0],
					correct / train_size
					))
				correct = 0
				train_size = 0

	def test(self):
		self.model.eval()
		test_loss = 0
		correct = 0
		flag = False
		if isinstance(self.criterion,nn.CrossEntropyLoss):
			self.criterion.size_average=False
			flag = True
		start_time = time()
		for data,label in self.test_loader:
			if self.cuda:
				data,label = data.cuda(),label.cuda()
			data,label = Variable(data,volatile=True),Variable(label)
			output = self.model(data)
			test_loss += self.criterion(output,label).data[0]
			pred = output.data.max(1,keepdim=True)[1]
			correct += pred.eq(label.data.view_as(pred)).cpu().sum()
		test_loss /= len(self.test_loader.dataset)
		print('\n Test_average_loss: {:.4f}, Acc: {}/{} ({:.1f}% Time: {:.4f}s)\n'.format(
			test_loss,
			correct,
			len(self.test_loader.dataset),
			100. * correct / len(self.test_loader.dataset),
			time() - start_time,
			))
		if flag:
			self.criterion.size_average=True
		return correct / float(len(self.test_loader.dataset))

	def start(self):
		print(self.model)
		if self.cuda:
			print('Using gpu devices:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
			self.model.cuda()
			# self.criterion.cuda()
		print('-----Start Training-----\n')
		best_precision = 0
		for e in range(self.start_epoch,self.epochs):
			if e in [self.epochs*0.5, self.epochs*0.75]:
				for param_group in self.optimizer.param_groups:
					param_group['lr'] *= 0.1
			self.train(e)
			precision = self.test()
			is_best = precision > best_precision
			training_state={
			'start_epoch': e + 1,
			'precision': precision,
			}
			self.save_checkpoint(
				training_state,
				is_best
				)
		print("-----Training Completed-----\n")

	def save_checkpoint(self,training_state,is_best):
		state = {
		'cuda': self.cuda,
		'start_epoch': training_state['start_epoch'],
		'epochs': self.epochs,
		'precision': training_state['precision'],
		'model_state_dict': self.model.state_dict(),
		'optimizer_state_dict': self.optimizer.state_dict()
		}
		file = os.path.join(self.root,'checkpoint.pkl')
		file_best = os.path.join(self.root,'model_best_checkpoint.pkl')
		torch.save(state,file)
		if is_best:
			shutil.copyfile(file,file_best)

	def load_checkpoint(self,root,is_resumed=False):
		if os.path.isfile(root):
			print("Loading checkpoint at '{}'".format(root))
			checkpoint = torch.load(root)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.cuda = checkpoint['cuda']
			if is_resumed:
				self.start_epoch = checkpoint['start_epoch']
				self.epochs = checkpoint['epochs']
				self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			return True
		else:
			print("'{}' doesn't exist!".format(root))
			return False

	def resume(self,root):
		if self.load_checkpoint(root,is_resumed=True):
			print('Successfully resume')
		else:
			print('Failed to resume')







def save_checkpoint(model,optimizer,training_state,is_best,root='checkpoint.pkl'):
	state = {
	'training_state': {
	'cuda': training_state['cuda'],
	'start_epoch': training_state['start_epoch'],
	'epochs': training_state['epochs'],
	'precision': training_state['precision'],
	},
	'model_state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict()
	}
	torch.save(state,root)
	if is_best:
		shutil.copyfile(root,'model_best_checkpoint.pkl')

def load_checkpoint(root,model,optimizer=None):
	if os.path.isfile(root):
		print("Loading checkpoint at '{}'".format(root))
		checkpoint = torch.load(root)
		training_state = checkpoint['training_state']
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		print('Loading completed!')
		return model,optimizer,training_state
	else:
		print("Checkpoint file doesn't exist at '{}'".format(root))
		return model,optimizer,{}

def train(model,train_loader,test_loader,optimizer,criterion,start_epoch=0,epochs=100,cuda=False,log_interval=100,**kwargs):
	model.train()
	best_precision = 0.
	print(model)
	if cuda:
		model.cuda()
	print('-----Start Training-----\n')
	for e in range(start_epoch,epochs):
		if e in [epochs*0.5, epochs*0.75]:
			for param_group in optimizer.param_groups:
				param_group['lr'] *= 0.1
		correct = 0
		train_size =0
		for batch_idx,(data,label) in enumerate(train_loader):
			if cuda:
				data,label = data.cuda(),label.cuda()
			data,label = Variable(data),Variable(label)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,label)
			loss.backward()
			optimizer.step()
			pred = output.data.max(1,keepdim=True)[1]
			correct += pred.eq(label.data.view_as(pred)).cpu().sum()
			train_size += len(data)
			if (batch_idx + 1) % log_interval == 0:
				print("Epoch: {} [{}/{} ({:.2f}%)]\t Loss: {:.6f} Acc: {:.6f}".format(
					e,
					(batch_idx + 1) * len(data),
					len(train_loader.dataset),
					100. * (batch_idx + 1) / len(train_loader),
					loss.data[0],
					correct / train_size
					))
				correct = 0
				train_size = 0
		precision = test(model,test_loader,criterion,cuda)
		is_best = precision > best_precision
		training_state={
		'cuda': cuda,
		'start_epoch': e + 1,
		'epochs': epochs,
		'precision': precision,
		}
		save_checkpoint(
			model,
			optimizer,
			training_state,
			is_best
			)
	print("-----Training Completed-----\n")

def test(model,test_loader,criterion,cuda=False,**kwargs):
	model.eval()
	test_loss = 0
	correct = 0
	criterion.size_average=False
	start_time = time()
	for data,label in test_loader:
		if cuda:
			data,label = data.cuda(),label.cuda()
		data,label = Variable(data,volatile=True),Variable(label)
		output = model(data)
		test_loss += criterion(output,label).data[0]
		pred = output.data.max(1,keepdim=True)[1]
		correct += pred.eq(label.data.view_as(pred)).cpu().sum()
	test_loss /= len(test_loader.dataset)
	print('\n Test_average_loss: {:.4f}, Acc: {}/{} ({:.1f}% Time: {:.4f}s)\n'.format(
		test_loss,
		correct,
		len(test_loader.dataset),
		100. * correct / len(test_loader.dataset),
		time() - start_time,
		))
	criterion.size_average=True
	return correct / float(len(test_loader.dataset))

# def resume(model,optimizer,criterion):
