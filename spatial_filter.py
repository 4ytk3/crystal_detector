from __future__ import annotations
import cv2
import numpy as np
from image_processor import Image
from abc import ABC, abstractmethod

class SpatialFilter(Image, ABC):
    def __init__(self, image: Image, *args):
        self._title = self.set_title(image._title, *args)
        self._gray_image = self.apply_filter(image._gray_image.copy(), *args)
        self._rgb_image = self.gray2rgb(self._gray_image)

    @abstractmethod
    def apply_filter(self):
        raise NotImplementedError()

class AverageFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3):
        super().__init__(image, ksize)

    def set_title(self, title: str, ksize: int, *args):
        return f"Average{ksize} " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3):
        return cv2.blur(src=gray_image, ksize=(ksize,ksize))

class GaussianFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3, sigmaX: int = 3):
        super().__init__(image, ksize, sigmaX)

    def set_title(self, title: str, ksize: int, *args):
        return f"Gaussian{ksize} " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3, sigmaX: int = 3):
        return cv2.GaussianBlur(src=gray_image, ksize=(ksize,ksize), sigmaX=sigmaX)

class MedianFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3):
        super().__init__(image, ksize)

    def set_title(self, title: str, ksize: int, *args):
        return f"Median{ksize} " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3):
        return cv2.medianBlur(src=gray_image, ksize=ksize)

class BilateralFilter(SpatialFilter):
    def __init__(self, image: Image, ksize: int = 3, sigmaColor=10, sigmaSpace=20):
        super().__init__(image, ksize, sigmaColor, sigmaSpace)

    def set_title(self, title: str, ksize: int, *args):
        return f"Bilateral{ksize} " + title.replace("Original ", "")

    def apply_filter(self, gray_image: np.ndarray, ksize: int = 3, sigmaColor=50, sigmaSpace=50):
        return cv2.bilateralFilter(src=np.float32(gray_image), d=ksize, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
