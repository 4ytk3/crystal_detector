from __future__ import annotations
import cv2
import numpy as np
from image_processor import Image, OriginalImage, ImageProcessor
from abc import ABC, abstractmethod

class SpatialFilter(Image, ABC):
    def __init__(self, image: Image, *args):
        self._title = self.set_title(image._title)
        self._gray_image = self.apply_filter(image._gray_image, *args)
        self._rgb_image = self.gray2rgb(self._gray_image)

    @abstractmethod
    def apply_filter(self):
        raise NotImplementedError()

class AverageFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3):
        super().__init__(image, ksize)

    def set_title(self, title: str):
        return "Average " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3):
        return cv2.blur(src=gray_image, ksize=(ksize,ksize))

class GaussianFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3, sigmaX: int = 3):
        super().__init__(image, ksize, sigmaX)

    def set_title(self, title: str):
        return "Gaussian " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3, sigmaX: int = 3):
        return cv2.GaussianBlur(src=gray_image, ksize=(ksize,ksize), sigmaX=sigmaX)

class MedianFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3):
        super().__init__(image, ksize)

    def set_title(self, title: str):
        return "Median " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3):
        return cv2.medianBlur(src=np.float32(gray_image), ksize=ksize)

class BilateralFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3, sigmaColor=10, sigmaSpace=10):
        super().__init__(image, ksize, sigmaColor, sigmaSpace)

    def set_title(self, title: str):
        return "Bilateral " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3, sigmaColor=10, sigmaSpace=10):
        return cv2.bilateralFilter(src=np.float32(gray_image), d=ksize, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


if __name__ == '__main__':
    title = "NaCl"
    path = "image/NaCl1_noscale.jpg"
    nacl = OriginalImage(title, path)
    average = AverageFilter(nacl)
    ImageProcessor.show_image(average)