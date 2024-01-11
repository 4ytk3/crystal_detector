import os
import sys
from typing import Any
import cv2
import copy
import numpy as np
from abc import ABC, abstractmethod
from image_processor import Image, OriginalImage, ImageProcessor
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max

class FFT(Image):
    def __init__(self, image: Image, *args):
        self._title = self.set_title(image._title)
        self._shifted_fft = self.get_fft(image._gray_image)
        self._fft_image = self.get_spectrum(self._shifted_fft)
        self._display_image = self.gray2rgb(self._fft_image)

    def set_title(self, title: str) -> str:
        return "FFT " + title.replace("Original ", "")

    def get_fft(self, gray_image):
        return np.fft.fftshift(np.fft.fft2(gray_image))

    def get_spectrum(self, shifted_fft):
        return 20 * np.log(np.abs(shifted_fft))

class IFFT(Image):
    def __init__(self, fft_image: FFT, *args):
        self._title = self.set_title(fft_image._title)
        self._ifft_image = self.get_ifft(fft_image._shifted_fft).real

    def set_title(self, title: str) -> str:
        return "IFFT " + title.replace("FFT", "")

    def get_ifft(self, shifted_fft):
        return np.abs(np.fft.ifft2(np.fft.fftshift(shifted_fft)))

class SpatialFilter(IFFT, ABC):
    def __init__(self, fft_image: FFT, *args):
        self._title = self.set_title(fft_image._title)
        shifted_fft = self.apply_mask(fft_image._shifted_fft)
        self._ifft_image = self.get_ifft(shifted_fft).real
        self._display_image = self.gray2rgb(self._ifft_image)

    def apply_mask(self, shifted_fft):
        height, width = shifted_fft.shape[0], shifted_fft.shape[1]
        mask = self.make_mask(height, width)
        return shifted_fft*mask

    @staticmethod
    def calc_center(self, height, width):
        return width//2, height//2

    @abstractmethod
    def make_mask(self, height, width):
        pass

    @staticmethod
    def make_mask(self, height, width):
        return np.ones([height, width], dtype=np.uint8)

        self._height, self._width = self._fft_image.shape[0], self._fft_image.shape[1]
        self._center_x, self._center_y = self.calc_center(self._height, self._width)
        self._mask = self.make_mask()
class LowpassFilter(IFFT):
    def __init__(self, title, image, inner_radius=90):
        self._inner_radius = inner_radius
        super().__init__(title, image)
        self._title = "Lowpass " + self._title

    def make_mask(self):
        self._mask = np.zeros([self._height, self._width], dtype=np.uint8)
        cv2.circle(self._mask, center=(self._center_x, self._center_y), radius=self._inner_radius, color=255, thickness=-1)

class HighpassFilter(FastFourierTransform):
    def __init__(self, title, image, outer_radius=80):
        self._outer_radius = outer_radius
        super().__init__(title, image)
        self._title = "Highpass " + self._title

    def make_mask(self):
        self._mask = np.ones([self._height, self._width], dtype=np.uint8)
        cv2.circle(self._mask, center=(self._center_x, self._center_y), radius=self._outer_radius, color=0, thickness=-1)

class BandpassFilter(FastFourierTransform):
    def __init__(self, title, image, outer_radius=100, inner_radius=70):
        self._outer_radius, self._inner_radius = outer_radius, inner_radius
        super().__init__(title, image)
        self._title = "Bandpass " + self._title

    def make_mask(self):
        self._mask = np.zeros([self._height, self._width], dtype=np.uint8)
        cv2.circle(self._mask, center=(self._center_x, self._center_y), radius=self._outer_radius, color=1, thickness=-1)
        cv2.circle(self._mask, center=(self._center_x, self._center_y), radius=self._inner_radius, color=0, thickness=-1)

class BandFilter(FastFourierTransform):
    def __init__(self, title, image, angle=0, size=10):
        self._angle = angle
        self._size = size
        super().__init__(title, image)
        self._title = "Band " + self._title

    def make_mask(self):
        self._square_mask = np.zeros([self._height, self._width], dtype=np.uint8)
        cv2.rectangle(self._square_mask, pt1=(int(self._width/2)-int(self._size/2),0), pt2=(int(self._width/2)+int(self._size/2),self._height), color=(255, 255, 255), thickness=-1)
        self.rotate_mask()
        self._mask = self._square_mask

    def rotate_mask(self):
        trans = cv2.getRotationMatrix2D(center=(self._center_x, self._center_y), angle=self._angle, scale=1.0)
        self._square_mask = cv2.warpAffine(self._square_mask, trans, (self._width, self._height))

class DoubleBandFilter(BandFilter):
    def __init__(self, title, image, angle=0, size=10, outer_radius=100, inner_radius=70):
        self._outer_radius, self._inner_radius = outer_radius, inner_radius
        super().__init__(title, image, angle, size)
        self._title = "Double " + self._title

    def make_mask(self):
        self._square_mask = np.zeros([self._height, self._width], dtype=np.uint8)
        cv2.rectangle(self._square_mask, pt1=(int(self._width/2)-int(self._size/2),0), pt2=(int(self._width/2)+int(self._size/2),self._height), color=(255, 255, 255), thickness=-1)
        self.rotate_mask()
        self._circle_mask = np.zeros([self._height, self._width], dtype=np.uint8)
        cv2.circle(self._circle_mask, center=(self._center_x, self._center_y), radius=self._outer_radius, color=1, thickness=-1)
        cv2.circle(self._circle_mask, center=(self._center_x, self._center_y), radius=self._inner_radius, color=0, thickness=-1)
        self._mask = self._square_mask * self._circle_mask

class PeakFilter(FastFourierTransform):
    def __init__(self, title, image, outer_radius=65, inner_radius=50):
        self._outer_radius, self._inner_radius = outer_radius, inner_radius
        super().__init__(title, image)
        self._title = "Peak " + self._title

    def make_mask(self):
        self._circle_mask = np.zeros([self._height, self._width], dtype=np.uint8)
        cv2.circle(self._circle_mask, center=(self._center_x, self._center_y), radius=self._outer_radius, color=1, thickness=-1)
        cv2.circle(self._circle_mask, center=(self._center_x, self._center_y), radius=self._inner_radius, color=0, thickness=-1)
        self._bandpass_image = self._fft_image*self._circle_mask
        #self.view_image(image=self._bandpass_image)
        self.detect_peaks()
        self.indices = np.dstack(np.where(self._bandpass_image == 1))
        for index in self.indices[0]:
            cv2.circle(self._bandpass_image, center=(index[1], index[0]), radius=5, color=1, thickness=-1)
        #self.view_image(image=self._bandpass_image)
        self._mask = self._bandpass_image

    def detect_peaks(self, filter_size=5, order=0.5):
        self.__local_max = maximum_filter(self._bandpass_image, footprint=np.ones((filter_size, filter_size)), mode='constant')
        #self.view_image(image=self.__local_max)
        self._bandpass_image[self.__local_max!=self._bandpass_image] = [0]
        self._bandpass_image[self._bandpass_image.max()!=self._bandpass_image] = [0]
        self._bandpass_image[self._bandpass_image!=0] = [1]
        #self.view_image(image=self._bandpass_image)
