import cv2
from PIL import Image
import numpy as np

def watershed_segmentation(image_file):
    image = Image.open(image_file.stream)
    np_image = np.asarray(image)[:,:,:3]
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    masked_image = cv2.bitwise_and(np_image, np_image, mask=mask) 
    return masked_image