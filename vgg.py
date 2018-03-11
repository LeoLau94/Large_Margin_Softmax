import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from l_softmax import L_Softmax

class vgg(nn.Module):

	def __init__(self,num_classes=10,margin=2,init_weight=True,cfg=None):
		super(vgg,self).__init__()
		if cfg is None:
			cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
		self.feature = self.make_layers(cfg,True)
		self.classifier = nn.Sequential(
			nn.Linear(cfg[-1],4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(True),
			nn.Dropout(),
			)
		self.l_softmax = L_Softmax(4096,num_classes,margin)
		if init_weight:
			self._initialize_weights()

	def make_layers(self,cfg,batch_norm=False):
		layers = []
		in_channels = 3
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1,bias=False)
				if batch_norm:
					layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
				else:
					layers += [conv2d,nn.ReLU(True)]
				in_channels = v
		return nn.Sequential(*layers)

	def forward(self,x,target=None):
		x = self.feature(x)
		x = nn.AvgPool2d(2)(x)
		x = x.view(x.size(0),-1)
		x = self.classifier(x)
		x = self.l_softmax(x,target)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0,math.sqrt(2./n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m,nn.BatchNorm2d):
				m.weight.data.fill_(0.5)
				m.bias.data.zero_()
			elif isinstance(m,nn.Linear):
				m.weight.data.normal_(0,0.01)
				m.bias.data.zero_()

if __name__ == '__main__':
	net = vgg()
	x = Variable(torch.FloatTensor(16,3,40,40))
	y = net(x)
	print(y.data.shape)