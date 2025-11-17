import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

IMG_DIR = 'images/'
IMG_FILES = ['img1.jpg', 'img2.jpg', 'img3.jpg']

os.makedirs("output", exist_ok=True)

for img_file in IMG_FILES:
    img_color = cv2.imread(IMG_DIR + img_file, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    
    totals = img_rgb.sum(axis=(0,1)).astype('float64') # 分别计算三个颜色通道的总和
    proportions = totals / totals.sum() # 计算各通道占比

    plt.figure(figsize=(5,3))
    labels = ['R', 'G', 'B']
    plt.bar(labels, proportions, color=['r','g','b'])
    for i, p in enumerate(proportions):
        plt.text(i, p + 0.01, f'{p*100:.2f}%', ha='center')
    plt.ylim(0, 1.0)
    plt.title(f'Color Histogram - {img_file}')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig(f'output/{os.path.splitext(img_file)[0]}_color.png')
    plt.close()
    
for img_file in IMG_FILES:
    img_gray = cv2.imread(IMG_DIR + img_file, cv2.IMREAD_GRAYSCALE) # 以灰度方式读入
    
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256]) # 直接绘制灰度直方图
    
    plt.figure()
    plt.plot(hist_gray, color='black')
    plt.xlim([0, 256])
    plt.title(f"Gray Histogram - {img_file}")
    plt.xlabel("Gray level")
    plt.ylabel("Frequency")
    plt.savefig(f"output/{os.path.splitext(img_file)[0]}_gray.png")
    plt.close()
    
    img_gray = img_gray.astype('float64')
    
    grad_x = np.zeros_like(img_gray, dtype=np.float64)
    grad_y = np.zeros_like(img_gray, dtype=np.float64)

    for y in range(1, img_gray.shape[0]-1): # 按PPT给定的方式计算梯度
        for x in range(1, img_gray.shape[1]-1):
            grad_x[y, x] = img_gray[y, x+1] - img_gray[y, x-1]
            grad_y[y, x] = img_gray[y+1, x] - img_gray[y-1, x]
            
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    hist_grad = cv2.calcHist([grad_mag.astype('float64')], [0], None, [361], [0, 361])
    
    plt.figure()
    plt.bar(range(361), hist_grad.ravel(), color='k', width=1)
    plt.xlim([-1, 361])
    plt.ylim(bottom=0)
    plt.title(f"Gradient Histogram - {img_file}")
    plt.xlabel("Gradient magnitude")
    plt.ylabel("Frequency")
    plt.savefig(f"output/{os.path.splitext(img_file)[0]}_gradient.png")
    plt.close()