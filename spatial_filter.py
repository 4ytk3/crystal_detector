import os
import sys
import cv2
import copy
import numpy as np
from
from abc import abstractclassmethod, ABC, ABCMeta, abstractmethod
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max

class SpatialFilter(ABC):
    def __init__(self, image: object):
        self._rgb_image = image._rgb_image
        self._gray_image = image._gray_image
        self._title = image._title

    @abstractmethod
    def apply_filter(self, *args):
        pass

class AverageFilter(SpatialFilter):
    def __init__(self, image: str, title: str):
        super().__init__(image, title)
        self._title = "Average " + self._title

    def apply_filter(self, ksize: int=3):
        self._process_image = cv2.blur(src=self._gray_image, ksize=(ksize,ksize))

class AverageFilter(SpatialFilter):
    def __init__(self, title, image, ksize=3):
        super().__init__(title, image)
        self._title = "Average " + self._title
        self._process_image = cv2.blur(src=self._gray_image, ksize=(ksize,ksize))

class GaussianFilter(SpatialFilter):
    def __init__(self, title, image, ksize=3, sigmaX=3):
        super().__init__(title, image)
        self._title = "Gaussian " + self._title
        self._process_image = cv2.GaussianBlur(src=self._gray_image, ksize=(ksize,ksize), sigmaX=sigmaX)

class MedianFilter(SpatialFilter):
    def __init__(self, title, image, ksize=3):
        super().__init__(title, image)
        self._title = "Median " + self._title
        self._process_image = cv2.medianBlur(src=np.float32(self._gray_image), ksize=ksize)

class BilateralFilter(SpatialFilter):
    def __init__(self, title, image, ksize=3, sigmaColor=10, sigmaSpace=10):
        super().__init__(title, image)
        self._title = "Bilateral " + self._title
        self._process_image = cv2.bilateralFilter(src=np.float32(self._gray_image), d=ksize, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
