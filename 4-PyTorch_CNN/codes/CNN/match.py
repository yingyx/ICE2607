# SJTU EE208

import time

import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

model_name = 'resnet50'
weights = 'ResNet50_Weights.IMAGENET1K_V1'

# Uncomment the below 2 lines to use alexnet
model_name = 'alexnet'
weights = 'AlexNet_Weights.IMAGENET1K_V1'

print('Model:', model_name)
model = torch.hub.load('pytorch/vision', model_name, weights=weights)

print(model)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

dataset_path = './dataset'
image_paths = []
for cls_id in os.listdir(dataset_path):
    for p in os.listdir(os.path.join(dataset_path, cls_id)):
        img_path = os.path.join(dataset_path, cls_id, p)
        if os.path.isfile(img_path):
            image_paths.append(img_path)

target_path = './target'            
target_paths = []
for p in os.listdir(target_path):
    img_path = os.path.join(target_path, p)
    if os.path.isfile(img_path):
        target_paths.append(img_path)

def features(x):
    if model_name == 'resnet50':
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
    elif model_name == 'alexnet':
        x = model.features(x)
        x = model.avgpool(x)
        
    return x

image_features = {}

for path in image_paths:
    image = default_loader(path)
    image = trans(image)
    image = torch.unsqueeze(image, 0)
    feature = features(image)
    feature = feature.detach().flatten()
    feature = feature / np.linalg.norm(feature)
    image_features[path] = feature
    
target_features = {}

for path in target_paths:
    image = default_loader(path)
    image = trans(image)
    image = torch.unsqueeze(image, 0)
    feature = features(image)
    feature = feature.detach().flatten()
    feature = feature / np.linalg.norm(feature)
    target_features[path] = feature

# for p1 in image_features:
#     sim = {}
#     for p2 in image_features:
#         if p1 == p2:
#             continue
#         sim[p2] = torch.dot(image_features[p1], image_features[p2])
#     nearest_paths = sorted(sim.keys(), key=lambda x:sim[x], reverse=True)
#     print('nearest of', p1, ':', nearest_paths[0], 'similarity', sim[nearest_paths[0]])

total = 0
total_top1 = 0
correct = 0
correct_top1 = 0
for p1 in target_features:
    sim = {}
    for p2 in image_features:
        if p1 == p2:
            continue
        sim[p2] = torch.dot(target_features[p1], image_features[p2])
    nearest_paths = sorted(sim.keys(), key=lambda x:sim[x], reverse=True)
    print('Matching', p1)
    for i in range(5):
        print('Top', i + 1, f'match ({sim[nearest_paths[i]].item()}):', nearest_paths[i])
        total += 1
        if i == 0:
            total_top1 += 1
        if os.path.split(p1)[1].split('.')[0] == os.path.split(os.path.split(nearest_paths[i])[0])[1]:
            correct += 1
            if i == 0:
                correct_top1 += 1
            
print('Accuracy (%):', 100. * correct / total)
print('Accuracy (top-1) (%):', 100. * correct_top1 / total_top1)