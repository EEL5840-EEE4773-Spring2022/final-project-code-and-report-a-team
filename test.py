# -*- coding: utf-8 -*-
'''===========================================Importing the dependencies==========================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
import os
import shutil


def test():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '1'     #If you are using GPU uncomment this line 
	#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	device = torch.device('cpu')
	'''====================================Loading the test data ====================================================='''

	parent_dir = os.getcwd()
	test_data_dir = os.path.join(parent_dir, 'test_data')

	if os.path.exists(test_data_dir):
		shutil.rmtree(test_data_dir)

	os.makedirs(test_data_dir)

	test_data_path = input("Specify the directory for 'data_test.npy' and 'labels_test.npy': \n")

	images_temp = np.load(test_data_path + '/' + 'data_test.npy')
	test_labels = np.load(test_data_path + '/' + 'labels_test.npy')

	num_images = len(test_labels)
	test_images_temp_T = images_temp.T
	test_images = test_images_temp_T.reshape(num_images, 300, 300)

	class_dir = []
	for p in range(10):
		class_dir.append('class_' + str(p))

	for d in range(len(class_dir)):
		os.makedirs(os.path.join(test_data_dir,class_dir[d]))
	i=0
	for i in range(int(len(test_images))):
		q = 0
		for q in range(len(class_dir)):
		    if str(test_labels[i]).split('.')[0] == class_dir[q].split('_')[1]:
		        im = Image.fromarray(test_images[i]) 
		        file_name = str(i) + '.jpg'
		        full_path = os.path.join(test_data_dir, os.path.join(class_dir[q], file_name)) 
		        im.save(full_path)
		    q+=1
		i+=1

	pretrained_size = 224
	pretrained_means = [0.485, 0.456, 0.406]
	pretrained_stds= [0.229, 0.224, 0.225]
	test_transforms = transforms.Compose([
		                       transforms.Resize(pretrained_size),
		                       transforms.CenterCrop(pretrained_size),
		                       transforms.ToTensor(),
		                       transforms.Normalize(mean = pretrained_means, 
		                                            std = pretrained_stds)
		                   ])
	test_data = datasets.ImageFolder(root = test_data_dir, transform = test_transforms)

	BATCH_SIZE = 12 
	test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)

	class_ids = ['a', 'b', 'c', 'd', 'e','f', 'g', 'h', '$', '#']
	print("Total number of test images: ", int(len(test_images)))

	#device = 'cpu'
	criterion = nn.CrossEntropyLoss()


	'''====================================Loading the trained model ====================================================='''
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

	#ResNet18
	resnet18_config = ResNetConfig(block = BasicBlock,
		                           n_blocks = [2,2,2,2],
		                           channels = [64, 128, 256, 512])
	#ResNet50
	resnet50_config = ResNetConfig(block = Bottleneck,
		                           n_blocks = [3, 4, 6, 3],
		                           channels = [64, 128, 256, 512])
	#ResNet34
	resnet34_config = ResNetConfig(block = BasicBlock,
		                           n_blocks = [3,4,6,3],
		                           channels = [64, 128, 256, 512])

	OUTPUT_DIM = len(test_data.classes)
	model = ResNet(resnet18_config, OUTPUT_DIM)
	#model = ResNet(resnet34_config, OUTPUT_DIM)
	#model = ResNet(resnet50_config, OUTPUT_DIM)
	model_path = input("Specify the full model path: \n") 
	model.load_state_dict(torch.load(model_path, map_location=device))

	'''========================================================Function Definitions======================================================='''

	def evaluate(model, iterator, criterion, device):
		
		loss_value = 0
		accuracy = 0 
		    
		model.eval()
		    
		with torch.no_grad():
		        
		    for (x, y) in iterator:

		        x = x.to(device)
		        y = y.to(device)

		        y_pred, _ = model(x)

		        loss = criterion(y_pred, y)

		        loss_value += loss.item()
		        
		        
		loss_value /= len(iterator)
		        
		return loss_value

	def get_predictions(model, iterator):

		model.eval()

		images = []
		labels = []
		probs = []

		with torch.no_grad():

		    for (x, y) in iterator:

		        x = x.to(device)

		        y_pred, _ = model(x)

		        y_prob = F.softmax(y_pred, dim = -1)
		        top_pred = y_prob.argmax(1, keepdim = True)

		        images.append(x.cpu())
		        labels.append(y.cpu())
		        probs.append(y_prob.cpu())

		images = torch.cat(images, dim = 0)
		labels = torch.cat(labels, dim = 0)
		probs = torch.cat(probs, dim = 0)

		return images, labels, probs 

	def plot_confusion_matrix(labels, pred_labels, classes):
		
		fig = plt.figure(figsize = (50, 50));
		ax = fig.add_subplot(1, 1, 1);
		cm = confusion_matrix(labels, pred_labels);
		cm = ConfusionMatrixDisplay(cm, display_labels = classes);
		cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
		plt.xticks(rotation = 90)
		plt.xlabel('Predicted Label', fontsize = 20)
		plt.ylabel('True Label', fontsize = 20)
		plt.savefig('Confusion_Matrix.png')
		#plt.show()

	'''==========================================================Running the code=============================================================='''

	test_loss = evaluate(model, test_iterator, criterion, device)


	imgs, lbs, prob_score = get_predictions(model, test_iterator)


	labels_predicted = torch.argmax(prob_score, 1)

	print(lbs)
	print(labels_predicted)

	plot_confusion_matrix(lbs, labels_predicted, class_ids) 

	labels_ = lbs.numpy()
	labels_pred = labels_predicted.numpy()

	n_authentic_pred =  (np.where(labels_pred==0))[0].size                    
	n_authentic_actual = (np.where(labels_==0))[0].size     
	n_counterfeit_pred = (np.where(labels_pred==1))[0].size     
	n_counterfeit_actual = (np.where(labels_==1))[0].size   
	test_accuracy = (accuracy_score(labels_, labels_pred))*100
	 
	print('Test Loss: {:.3f} | Test Accuracy: {:6.2f}%\n'.format(test_loss, test_accuracy))
	print('Actual Labels:\n')
	print(labels_)
	print('\nPredicted Labels:\n')
	print(labels_pred)
	print('\nClassification Report:\n')
	print(classification_report(labels_, labels_pred, target_names=class_ids))
	print('\nCode ran successfully.')

#Invoking the test module
test()
