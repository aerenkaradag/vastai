import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
from torchvision import transforms
import xml.etree.ElementTree as ET

# Hyper Parameters
FEATURE_DIM = 64
RELATION_DIM = 8
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 10
EPISODE = 10
TEST_EPISODE = 600
LEARNING_RATE = 0.001
GPU = 0
HIDDEN_UNIT = 10
transform = transforms.Compose([transforms.Resize((84,84)), 
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])])        

def get_bbox(file_path, object_name):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == object_name:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            width = xmax 
            height = ymax 

            return xmin, ymin, xmax, ymax

    return None  # If the object with the specified name is not foundth the specified name is not found

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out 

class RelationNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

class OneShotLearning:
    def __init__(self):
        self.sample_features = None
        self.feature_encoder = CNNEncoder()
        self.relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)
        self.feature_encoder.cuda(GPU)
        self.relation_network.cuda(GPU)
        self.test_features = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_encoder.load_state_dict(torch.load(str("defrcn/evaluation/models/miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"), map_location=device))
        self.relation_network.load_state_dict(torch.load(str("defrcn/evaluation/models/miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"),  map_location=device))
    def find_relations(self):
        
        sample_features_ext = self.sample_features.unsqueeze(0).repeat(5,1,1,1,1)
        test_features_ext = self.test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        test_features_ext = torch.transpose(test_features_ext,0,1)

        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
        relations = self.relation_network(relation_pairs).view(-1,CLASS_NUM)
        relations = relations[:,0]
        return relations
    def initialize_sample(self):
      sample_images = []
      img_pth = "datasets/VOC2007/JPEGImages/009975.jpg"
      sample_image = Image.open(img_pth)
      sample_image = sample_image.convert('RGB')
      
      bbo = get_bbox("datasets/VOC2007/Annotations/009975.xml", "secondnew")
      print(bbo)
      for i in range(5):
        xmin, ymin, xmax, ymax = bbo
        cropped_image = sample_image.crop((xmin, ymin, xmax, ymax))  # Crop image using bounding box
        cropped_image = transform(cropped_image).unsqueeze(0)  # Transform and add batch dimension
        sample_images.append(cropped_image)

      sample_images = torch.cat(sample_images, 0)  # Stack all sample images
      self.sample_features = self.feature_encoder(Variable(sample_images).cuda(GPU))
      self.sample_features = self.sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
      self.sample_features = torch.sum(self.sample_features,1).squeeze(1)
    def initialize_test(self,bboxes, pth):
      test_images = []
      test_image = Image.open(pth)
      test_image = test_image.convert('RGB')
      for bbox in bboxes:
        xmin, ymin, width, height = bbox
        cropped_image = test_image.crop((xmin, ymin, xmin+width, ymin+height))  # Crop image using bounding box
        cropped_image = transform(cropped_image).unsqueeze(0)  # Transform and add batch dimension
        test_images.append(cropped_image)
      test_images = torch.cat(test_images, 0)  # Stack all test images
      self.test_features = self.feature_encoder(Variable(test_images).cuda(GPU))