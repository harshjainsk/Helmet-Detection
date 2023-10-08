import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2


model_path = os.path.join("E:/wobot-git/runs/train/exp/weights/last.pt")
model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path, force_reload = True)

image_path = os.path.join('E:/wobot-git/testing_images/test_image_4.png')

img = cv2.imread("testing_images/test_image_4.png")
img = cv2.resize(img, (640,640))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("img", img)

# %matplotlib inline

results = model(image_path)
results.print()

img = np.squeeze(results.render())
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("img", img)
cv2.waitKey(0)
