import cv2
import numpy as np
from image_processor import Image

class CannyEdgeDetector(Image):
    def __init__(self, image: Image, min_val=70, max_val=150):
        self._title = self.set_title(image._title)
        self._gray_image = self.canny_edge(image._gray_image, min_val, max_val)

    def set_title(self, title: str):
        return "Canny " + title.replace("Original ", "")

    def canny_edge(self, gray_image, min_val, max_val):
        return cv2.Canny(gray_image.astype(np.uint8), threshold1=min_val, threshold2=max_val, L2gradient=True)

class SobelEdgeDetector(Image):
    def __init__(self, image: Image, dx=1, dy=1, ksize=3, amp=3):
        self._title = self.set_title(image._title)
        self._gray_image = self.sobel_edge(image._gray_image, dx, dy, ksize, amp)

    def set_title(self, title: str):
        return "Sobel " + title.replace("Original ", "")

    def sobel_edge(self, gray_image, dx, dy, ksize, amp):
        return cv2.convertScaleAbs(cv2.Sobel(gray_image.astype(np.uint8), cv2.CV_8U, dx, dy, ksize=ksize)) * amp

class LaplacianEdgeDetector(Image):
    def __init__(self, image: Image, amp=10):
        self._title = self.set_title(image._title)
        self._gray_image = self.laplacian_edge(image._gray_image, amp)

    def set_title(self, title: str):
        return "Laplacian " + title.replace("Original ", "")

    def laplacian_edge(self, gray_image, amp):
        return cv2.convertScaleAbs(cv2.Laplacian(gray_image.astype(np.uint8), cv2.CV_8U)) * amp