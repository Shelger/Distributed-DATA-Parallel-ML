import cv2
import numpy as np
from PIL import Image, ImageEnhance
from itertools import product

def HSV_mask(img_hsv, lower):
    lower = np.array(lower)
    upper = np.array([255, 255, 255])
    return cv2.inRange(img_hsv, lower, upper)

def preprocess_image(input_path, output_path):
    img = cv2.imread(input_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray[img_gray >= 235] = 255
    mask1 = HSV_mask(img_hsv, [0, 0, 155])[..., None].astype(np.float32)
    mask2 = HSV_mask(img_hsv, [0, 20, 0])
    masked = np.uint8((img + mask1) / (1 + mask1 / 255))
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray[gray >= 180] = 255
    gray[mask2 == 0] = img_gray[mask2 == 0]
    processed_image = np.clip(1.80*gray-35, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, processed_image)
    return processed_image
