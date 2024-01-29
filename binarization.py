import cv2
import numpy as np
from image_processor import Image

class OtsuBinarization(Image):
    def __init__(self, image: Image, threshold=128):
        self._title = self.set_title(image._title)
        ret, self._gray_image = self.otsu_bin(image._gray_image.astype("uint8"), threshold)

    def set_title(self, title: str):
        return "Binarization " + title.replace("Original ", "")

    def otsu_bin(self, gray_image: np.ndarray, threshold):
        return cv2.threshold(gray_image, threshold, 255, cv2.THRESH_OTSU)

class AdaptiveBinarization(Image):
    def __init__(self, image: Image, blockSize=5):
        self._title = self.set_title(image._title)
        self._gray_image = self.adapt_bin(image._gray_image, blockSize)

    def set_title(self, title: str):
        return "Binarization " + title.replace("Original ", "")

    def adapt_bin(self, gray_image: np.ndarray, blockSize=5):
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=blockSize, C=0)
