# -*- coding: utf-8 -*-
'''=================================Importing the dependencies==================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import copy
from collections import namedtuple
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import shutil
import time

def train():
	'''=================================Setting the random seeds======================================'''
	SEED = 2022

	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True

	'''=================================Hyperparameters==============================================='''
	START_LR = 1e-7
	END_LR = 10
	NUM_ITER = 100
	BATCH_SIZE = 12
	EPOCHS = 20  #5, 10, 50, 100, 200

	'''=================================Dataloader and Utilities======================================'''
	root = os.getcwd()
	images_dir = os.path.join(root, 'images')
	train_dir = os.path.join(root, 'train')
	val_dir = os.path.join(root, 'valid')


	if os.path.exists(images_dir):
		shutil.rmtree(images_dir) 
	if os.path.exists(train_dir):
		shutil.rmtree(train_dir) 
	if os.path.exists(val_dir):
		shutil.rmtree(val_dir)

	os.makedirs(images_dir)
	os.makedirs(train_dir)
	os.makedirs(val_dir) 



	dirname = input("Specify the directory for 'data_train.npy' and 'labels_train.npy': \n")

	images_temp = np.load(dirname + '/' + 'data_train.npy')
	labels = np.load(dirname + '/' + 'labels_train.npy')

	num_images = len(labels)
	images_temp_T = images_temp.T
	images = images_temp_T.reshape(num_images, 300, 300)

	print("Total number of images: ", int(len(labels)))
	class_dir = []
	for p in range(10):
		class_dir.append('class_' + str(p))

	for d in range(len(class_dir)):
		os.makedirs(os.path.join(images_dir,class_dir[d]))
	i=0
	for i in range(int(len(images))):
		q = 0
		for q in range(len(class_dir)):
		    if str(labels[i]).split('.')[0] == class_dir[q].split('_')[1]:
		        im = Image.fromarray(images[i]) 
		        file_name = str(i) + '.jpg'
		        full_path = os.path.join(images_dir, os.path.join(class_dir[q], file_name)) 
		        im.save(full_path)
		    q+=1
		i+=1

	classes = os.listdir(images_dir)

	TRAIN_RATIO = 0.8

	for c in classes:
		
		class_dir = os.path.join(images_dir, c)
		
		images = os.listdir(class_dir)
		   
		n_train = int(len(images) * TRAIN_RATIO)
		
		train_images = images[:n_train]
		val_images = images[n_train:]
		
		os.makedirs(os.path.join(train_dir, c), exist_ok = True)
		os.makedirs(os.path.join(val_dir, c), exist_ok = True)
		
		for image in train_images:
		    image_src = os.path.join(class_dir, image)
		    image_dst = os.path.join(train_dir, c, image) 
		    shutil.copyfile(image_src, image_dst)
		    
		for image in val_images:
		    image_src = os.path.join(class_dir, image)
		    image_dst = os.path.join(val_dir, c, image) 
		    shutil.copyfile(image_src, image_dst)


	pretrained_size = 224
	pretrained_means = [0.485, 0.456, 0.406]
	pretrained_stds= [0.229, 0.224, 0.225]

	train_transforms = transforms.Compose([
		                       transforms.Resize(pretrained_size),
		                       transforms.RandomRotation(5),
		                       transforms.RandomHorizontalFlip(0.5),
		                       transforms.RandomCrop(pretrained_size, padding = 10),
		                       transforms.ToTensor(),
		                       transforms.Normalize(mean = pretrained_means, 
		                                            std = pretrained_stds)
		                   ])

	val_transforms = transforms.Compose([
		                       transforms.Resize(pretrained_size),
		                       transforms.CenterCrop(pretrained_size),
		                       transforms.ToTensor(),
		                       transforms.Normalize(mean = pretrained_means, 
		                                            std = pretrained_stds)
		                   ])

	image_data = datasets.ImageFolder(root = images_dir, 
		                              transform = train_transforms)

	train_data = datasets.ImageFolder(root = train_dir, 
		                              transform = train_transforms)

	val_data = datasets.ImageFolder(root = val_dir, 
		                             transform = val_transforms)



	n_train_examples = int(len(train_data))
	n_valid_examples = int(len(image_data))- n_train_examples

	train_data, valid_data = data.random_split(image_data, 
		                                       [n_train_examples, n_valid_examples])


	print("Total number of training images: ", int(len(train_data)))
	print("Total number of validation images: ",int(len(valid_data)))


	

	train_iterator = data.DataLoader(train_data, 
		                             shuffle = True, 
		                             batch_size = BATCH_SIZE)

	valid_iterator = data.DataLoader(valid_data, 
		                             batch_size = BATCH_SIZE)

	def normalize_image(image):
		image_min = image.min()
		image_max = image.max()
		image.clamp_(min = image_min, max = image_max)
		image.add_(-image_min).div_(image_max - image_min + 1e-5)
		return image

	def plot_images(images, labels, normalize = True):

		n_images = len(images)

		rows = int(np.sqrt(n_images))
		cols = int(np.sqrt(n_images))

		fig = plt.figure(figsize = (15, 15))

		for i in range(rows*cols):

		    ax = fig.add_subplot(rows, cols, i+1)
		    
		    image = images[i]

		    if normalize:
		        image = normalize_image(image)

		    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
		    label = labels[i]
		    ax.set_title(label)
		    ax.axis('off')
		plt.savefig('Images.png')
		#plt.show()
	N_IMAGES = 16

	images, labels = zip(*[(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]])

	classes = val_data.classes

	plot_images(images, labels, classes)


	class ResNet(nn.Module):
		def __init__(self, config, output_dim):
		    super().__init__()
		            
		    block, n_blocks, channels = config
		    self.in_channels = channels[0]
		        
		    assert len(n_blocks) == len(channels) == 4
		    
		    self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
		    self.bn1 = nn.BatchNorm2d(self.in_channels)
		    self.relu = nn.ReLU(inplace = True)
		    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
		    
		    self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
		    self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
		    self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
		    self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
		    
		    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		    self.fc = nn.Linear(self.in_channels, output_dim)
		    
		def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
		
		    layers = []
		    
		    if self.in_channels != block.expansion * channels:
		        downsample = True
		    else:
		        downsample = False
		    
		    layers.append(block(self.in_channels, channels, stride, downsample))
		    
		    for i in range(1, n_blocks):
		        layers.append(block(block.expansion * channels, channels))

		    self.in_channels = block.expansion * channels
		        
		    return nn.Sequential(*layers)
		    
		def forward(self, x):
		    
		    x = self.conv1(x)
		    x = self.bn1(x)
		    x = self.relu(x)
		    x = self.maxpool(x)

		    x = self.layer1(x)
		    x = self.layer2(x)
		    x = self.layer3(x)
		    x = self.layer4(x)
		    
		    x = self.avgpool(x)
		    h = x.view(x.shape[0], -1)
		    x = self.fc(h)
		    
		    return x, h

	class BasicBlock(nn.Module):
		
		expansion = 1
		
		def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
		    super().__init__()
		            
		    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
		                           stride = stride, padding = 1, bias = False)
		    self.bn1 = nn.BatchNorm2d(out_channels)
		    
		    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
		                           stride = 1, padding = 1, bias = False)
		    self.bn2 = nn.BatchNorm2d(out_channels)
		    
		    self.relu = nn.ReLU(inplace = True)
		    
		    if downsample:
		        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
		                         stride = stride, bias = False)
		        bn = nn.BatchNorm2d(out_channels)
		        downsample = nn.Sequential(conv, bn)
		    else:
		        downsample = None
		    
		    self.downsample = downsample
		    
		def forward(self, x):
		    
		    i = x
		    
		    x = self.conv1(x)
		    x = self.bn1(x)
		    x = self.relu(x)
		    
		    x = self.conv2(x)
		    x = self.bn2(x)
		    
		    if self.downsample is not None:
		        i = self.downsample(i)
		                    
		    x += i
		    x = self.relu(x)
		    
		    return x

	ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

	resnet18_config = ResNetConfig(block = BasicBlock,
		                           n_blocks = [2,2,2,2],
		                           channels = [64, 128, 256, 512])

	resnet34_config = ResNetConfig(block = BasicBlock,
		                           n_blocks = [3,4,6,3],
		                           channels = [64, 128, 256, 512])

	class Bottleneck(nn.Module):
		
		expansion = 4
		
		def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
		    super().__init__()
		
		    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
		                           stride = 1, bias = False)
		    self.bn1 = nn.BatchNorm2d(out_channels)
		    
		    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
		                           stride = stride, padding = 1, bias = False)
		    self.bn2 = nn.BatchNorm2d(out_channels)
		    
		    self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
		                           stride = 1, bias = False)
		    self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
		    
		    self.relu = nn.ReLU(inplace = True)
		    
		    if downsample:
		        conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
		                         stride = stride, bias = False)
		        bn = nn.BatchNorm2d(self.expansion * out_channels)
		        downsample = nn.Sequential(conv, bn)
		    else:
		        downsample = None
		        
		    self.downsample = downsample
		    
		def forward(self, x):
		    
		    i = x
		    
		    x = self.conv1(x)
		    x = self.bn1(x)
		    x = self.relu(x)
		    
		    x = self.conv2(x)
		    x = self.bn2(x)
		    x = self.relu(x)
		    
		    x = self.conv3(x)
		    x = self.bn3(x)
		            
		    if self.downsample is not None:
		        i = self.downsample(i)
		        
		    x += i
		    x = self.relu(x)
		
		    return x

	resnet50_config = ResNetConfig(block = Bottleneck,
		                           n_blocks = [3, 4, 6, 3],
		                           channels = [64, 128, 256, 512])

	resnet101_config = ResNetConfig(block = Bottleneck,
		                            n_blocks = [3, 4, 23, 3],
		                            channels = [64, 128, 256, 512])

	resnet152_config = ResNetConfig(block = Bottleneck,
		                            n_blocks = [3, 8, 36, 3],
		                            channels = [64, 128, 256, 512])

	pretrained_model = models.resnet18(pretrained = True)
	#pretrained_model = models.resnet34(pretrained = True)
	#pretrained_model = models.resnet50(pretrained = True)

	print(pretrained_model)

	IN_FEATURES = pretrained_model.fc.in_features 
	OUTPUT_DIM = len(val_data.classes)

	fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

	pretrained_model.fc = fc

	model = ResNet(resnet18_config, OUTPUT_DIM)
	#model = ResNet(resnet34_config, OUTPUT_DIM)
	#model = ResNet(resnet50_config, OUTPUT_DIM)

	model.load_state_dict(pretrained_model.state_dict())

	def count_parameters(model):
		return sum(p.numel() for p in model.parameters() if p.requires_grad)

	print("The model has {} trainable parameters".format(count_parameters(model)))



	optimizer = optim.Adam(model.parameters(), lr=START_LR)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	criterion = nn.CrossEntropyLoss()

	model = model.to(device)
	criterion = criterion.to(device)

	class LRFinder:
		def __init__(self, model, optimizer, criterion, device):
		    
		    self.optimizer = optimizer
		    self.model = model
		    self.criterion = criterion
		    self.device = device
		    
		    torch.save(model.state_dict(), 'init_params.pt')

		def range_test(self, iterator, end_lr = 10, num_iter = 100, 
		               smooth_f = 0.05, diverge_th = 5):
		    
		    lrs = []
		    losses = []
		    best_loss = float('inf')

		    lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
		    
		    iterator = IteratorWrapper(iterator)
		    
		    for iteration in range(num_iter):

		        loss = self._train_batch(iterator)

		        #update lr
		        lr_scheduler.step()
		        
		        lrs.append(lr_scheduler.get_lr()[0])

		        if iteration > 0:
		            loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
		            
		        if loss < best_loss:
		            best_loss = loss

		        losses.append(loss)
		        
		        if loss > diverge_th * best_loss:
		            print("Stopping early, the loss has diverged")
		            break
		                   
		    #reset model to initial parameters
		    model.load_state_dict(torch.load('init_params.pt'))
		                
		    return lrs, losses

		def _train_batch(self, iterator):
		    
		    self.model.train()
		    
		    self.optimizer.zero_grad()
		    
		    x, y = iterator.get_batch()
		    
		    x = x.to(self.device)
		    y = y.to(self.device)
		    
		    y_pred, _ = self.model(x)
		            
		    loss = self.criterion(y_pred, y)
		    
		    loss.backward()
		    
		    self.optimizer.step()
		    
		    return loss.item()

	class ExponentialLR(_LRScheduler):
		def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
		    self.end_lr = end_lr
		    self.num_iter = num_iter
		    super(ExponentialLR, self).__init__(optimizer, last_epoch)

		def get_lr(self):
		    curr_iter = self.last_epoch + 1
		    r = curr_iter / self.num_iter
		    return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

	class IteratorWrapper:
		def __init__(self, iterator):
		    self.iterator = iterator
		    self._iterator = iter(iterator)

		def __next__(self):
		    try:
		        inputs, labels = next(self._iterator)
		    except StopIteration:
		        self._iterator = iter(self.iterator)
		        inputs, labels, _ = next(self._iterator)

		    return inputs, labels

		def get_batch(self):
		    return next(self)




	lr_finder = LRFinder(model, optimizer, criterion, device)
	lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)


	def final_LR(LRs, Losses):
		for i in range(len(LRs)):
		    if Losses[i] == min(Losses):
		       final_lr = LRs[i]
		    else:
		       i+=1
		return final_lr


	def plot_lr_finder(lrs, losses, skip_start = 5, skip_end = 5):
		
		if skip_end == 0:
		    lrs = lrs[skip_start:]
		    losses = losses[skip_start:]
		else:
		    lrs = lrs[skip_start:-skip_end]
		    losses = losses[skip_start:-skip_end]
		
		fig = plt.figure(figsize = (16,8))
		ax = fig.add_subplot(1,1,1)
		ax.plot(lrs, losses)
		ax.set_xscale('log')
		ax.set_xlabel('Learning rate')
		ax.set_ylabel('Loss')
		ax.grid(True, 'both', 'x')
		plt.savefig('LR.png')
		#plt.show()

	plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)

	print("Final learning rate is: ",format(final_LR(lrs, losses),'.4E'))

	FOUND_LR = final_LR(lrs, losses) 

	params = [
		      {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
		      {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
		      {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
		      {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
		      {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
		      {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
		      {'params': model.fc.parameters()}
		     ]


	optimizer = optim.Adam(params, lr = FOUND_LR)


	STEPS_PER_EPOCH = len(train_iterator)
	TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

	MAX_LRS = [p['lr'] for p in optimizer.param_groups]

	scheduler = lr_scheduler.OneCycleLR(optimizer,
		                                max_lr = MAX_LRS,
		                                total_steps = TOTAL_STEPS)

	def calculate_topk_accuracy(y_pred, y, k = 5):
		with torch.no_grad():
		    batch_size = y.shape[0]
		    _, top_pred = y_pred.topk(k, 1)
		    top_pred = top_pred.t()
		    correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
		    correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
		    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
		    acc_1 = correct_1 / batch_size
		    acc_k = correct_k / batch_size
		return acc_1, acc_k

	def train(model, iterator, optimizer, criterion, scheduler, device):
		
		epoch_loss = 0
		epoch_acc_1 = 0
		epoch_acc_5 = 0
		
		model.train()
		
		for (x, y) in iterator:
		    
		    x = x.to(device)
		    y = y.to(device)
		    
		    optimizer.zero_grad()
		            
		    y_pred, _ = model(x)
		    
		    loss = criterion(y_pred, y)
		    
		    acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
		    
		    loss.backward()
		    
		    optimizer.step()
		    
		    scheduler.step()
		    
		    epoch_loss += loss.item()
		    epoch_acc_1 += acc_1.item()
		    epoch_acc_5 += acc_5.item()
		    
		epoch_loss /= len(iterator)
		epoch_acc_1 /= len(iterator)
		epoch_acc_5 /= len(iterator)
		    
		return epoch_loss, epoch_acc_5

	def evaluate(model, iterator, criterion, device):
		
		epoch_loss = 0
		epoch_acc_1 = 0
		epoch_acc_5 = 0
		    
		model.eval()
		    
		with torch.no_grad():
		        
		    for (x, y) in iterator:

		        x = x.to(device)
		        y = y.to(device)

		        y_pred, _ = model(x)

		        loss = criterion(y_pred, y)

		        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

		        epoch_loss += loss.item()
		        epoch_acc_1 += acc_1.item()
		        epoch_acc_5 += acc_5.item()
		        
		epoch_loss /= len(iterator)
		epoch_acc_1 /= len(iterator)
		epoch_acc_5 /= len(iterator)
		epoch_accuracy = epoch_acc_5
		        
		return epoch_loss, epoch_accuracy 

	def epoch_time(start_time, end_time):
		elapsed_time = end_time - start_time
		elapsed_mins = int(elapsed_time / 60)
		elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
		return elapsed_mins, elapsed_secs

	def plot_loss_curve(epochs, train_loss, val_loss):
		loss_train = train_loss
		loss_val =  val_loss
		Epochs = range(1, epochs+1)
		fig = plt.figure(figsize = (16,8))
		ax = fig.add_subplot(1,1,1)
		ax.plot(Epochs, loss_train, 'g', label='Training loss')
		ax.plot(Epochs, loss_val, 'b', label='Validation loss')
		ax.set_title('Training and Validation loss')
		ax.set_xlabel('Epochs')
		ax.set_ylabel('Loss')
		ax.grid(True, 'both', 'x')
		ax.legend()
		plt.savefig('loss_curve.png')
		#plt.show()

	def plot_accuracy_curve(epochs, train_accuracy, val_accuracy):
		accuracy_train = train_accuracy
		accuracy_val =  val_accuracy
		Epochs = range(1, epochs+1)

		fig = plt.figure(figsize = (16,8))
		ax = fig.add_subplot(1,1,1)
		ax.plot(Epochs, train_accuracy, 'g', label='Training accuracy')
		ax.plot(Epochs, val_accuracy, 'b', label='Validation accuracy')
		ax.set_title('Training and Validation accuracy')
		ax.set_xlabel('Epochs')
		ax.set_ylabel('Accuracy')
		ax.grid(True, 'both', 'x')
		ax.legend()
		plt.savefig('accuracy_curve.png')
		#plt.show()


	best_valid_loss = float('inf')
	train_loss_list = []
	val_loss_list = []
	train_acc_list = []
	val_acc_list = []

	for epoch in range(EPOCHS):
		
		start_time = time.monotonic()
		
		train_loss, train_acc = train(model, train_iterator, optimizer, criterion, scheduler, device)
		valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
		    
		if valid_loss < best_valid_loss:
		    best_valid_loss = valid_loss
		    torch.save(model.state_dict(), 'best_model_resnet18_e5.pt') #  'best_model_resnet18.pt', 'best_model_resnet34.pt', 'best_model_resnet50.pt'
		end_time = time.monotonic()

		epoch_mins, epoch_secs = epoch_time(start_time, end_time)

		train_loss_list.append(train_loss)
		val_loss_list.append(valid_loss)
		train_acc_list.append(train_acc)
		val_acc_list.append(valid_acc)
		
		print('Epoch: {} | Epoch Time: {}m {}s'.format((epoch+1),epoch_mins,epoch_secs))
		print('\tTrain Loss: {:.3f} | Train Accuracy: {:6.2f}% \n'.format(train_loss,train_acc*100))
		print('\tValid Loss: {:.3f} | Valid Accuracy: {:6.2f}% \n'.format(valid_loss, valid_acc*100))

	plot_loss_curve(EPOCHS, train_loss_list, val_loss_list)
	plot_accuracy_curve(EPOCHS, train_acc_list, val_acc_list)

#Calling the train module
train()
