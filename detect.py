import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'resnet18.pt')  # load from PyTorch Hub

#load images
im2 = cv2.imread(r'C:\Users\ranji\OneDrive\Desktop\yolov5-master\yolov5_new\source\person.png')

# Inference
results = model(im2) # batch of images
#results show
results.show()
#results write
# results.write()