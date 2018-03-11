import torch
from torch import nn
from torch.autograd import Variable
import math
from scipy.special import binom


class L_Softmax(nn.Module):
	def __init__(self,input_features,output_features,margin):
		super(L_Softmax,self).__init__()
		self.input_features = input_features
		self.output_features = output_features
		self.margin = margin
		self.weight = nn.Parameter(torch.FloatTensor(input_features,output_features))
		self.binom = binom(margin,list(range(0,margin+1,2)))
		self.cos_exp = list(range(self.margin, -1, -2))
		self.sin_exp = list(range(len(self.cos_exp)))
		self.signs = [1]
		for i in range(1,len(self.sin_exp)):
			self.signs.append(self.signs[-1]*-1)

	def rest_parameters(self):
		nn.init.kaiming_normal(self.weight.data.t())

	def find_k(self,cos):
		acos = cos.acos()
		k = (acos*self.margin/math.pi).floor().detach()
		return k

	def forward(self,x,target=None):
		if self.training:
			assert target is not None
			fy_i = x.matmul(self.weight)
			batch_size = fy_i.size(0)
			fy_i_target = fy_i[list(range(batch_size)),target.data]
			weight_target_norm = self.weight[:,target.data].norm(p=2,dim=0)
			x_norm = x.norm(p=2,dim=1)
			norm_mul = weight_target_norm*x_norm
			cos = fy_i_target/norm_mul
			sin = 1-cos**2
			k = self.find_k(cos)
			num_ns = self.margin//2 + 1
			binom = Variable(x.data.new(self.binom))
			cos_exp = Variable(x.data.new(self.cos_exp))
			sin_exp = Variable(x.data.new(self.sin_exp))
			signs = Variable(x.data.new(self.signs))
			cos_terms = cos.unsqueeze(1)**cos_exp.unsqueeze(0)
			sin_tems = sin.unsqueeze(1)**sin_exp.unsqueeze(0)
			cosm_terms = (signs.unsqueeze(0)*binom.unsqueeze(0)*cos_terms*sin_tems)
			cosm = cosm_terms.sum(1)
			fy_i_target = norm_mul * (((-1)**k * cosm) - 2*k)
			fy_i[list(range(batch_size)), target.data] = fy_i_target
			return fy_i
		else:
			assert target is None
			return x.matmul(self.weight)