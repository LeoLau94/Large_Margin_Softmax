import os
import argparse
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import Trainer
from vgg import vgg
from model import MNISTModel

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--fine-tune', default='', type=str, metavar='PATH',
                    help='fine-tune from pruned model')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--gpu-devices',type=str,default='0',help='decide which gpu devices to use.For exmaple:0,1')
parser.add_argument('--root',type=str,default='./', metavar='PATH', help='path to save checkpoint')
parser.add_argument('--margin', type=int, default=1, metavar='M',
                    help='set margin')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
     torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
     train_loader = torch.utils.data.DataLoader(
          datasets.CIFAR10('../data',train=True,download=False,
               transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
                    ])
               ),batch_size=args.batch_size,shuffle=True,**kwargs
          )
     test_loader = torch.utils.data.DataLoader(
          datasets.CIFAR10('../data',train=False,
               transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
                    ])
               ),
          batch_size = args.test_batch_size,shuffle=True,**kwargs
          )
elif args.dataset == 'cifar100':
     train_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100('../data',train=True,download=True,
               transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
                    ])
               ),batch_size=args.batch_size,shuffle=True,**kwargs
          )
     test_loader = torch.utils.data.DataLoader(
          datasets.CIFAR100('../data',train=False,
               transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((.5,.5,.5),(.5,.5,.5))
                    ])
               ),
          batch_size = args.test_batch_size,shuffle=True,**kwargs
          )

model = MNISTModel(margin=args.margin)
optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

print('\nNormal Training \n')
trainer = Trainer(
     model=model,
     optimizer=optimizer,
     criterion=criterion,
     start_epoch=args.start_epoch,
     epochs=args.epochs,
     cuda=args.cuda,
     log_interval=args.log_interval,
     train_loader=train_loader,
     test_loader=test_loader,
     root=args.root,
     )
trainer.start()
