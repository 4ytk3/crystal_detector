import os
import sys
import cv2
import copy
import numpy as np
from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt
import image_viewer
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max

class FastFourierTransform(image_viewer.Image):
    def __init__(self, title, image):
        super().__init__(title, image)
        self.get_fft()
        self.get_spectrum()
        self.make_mask()
        self.get_ifft()

    def get_fft(self):
        self._shifted_fft = np.fft.fftshift(np.fft.fft2(self._gray_image))

    def get_spectrum(self):
        self._fft_image = 20 * np.log(np.abs(self._shifted_fft))
        self._height, self._width = self._fft_image.shape[0], self._fft_image.shape[1]
        self._center_x, self._center_y = self._width//2, self._height//2
        self._mask = np.ones([self._height, self._width], dtype=np.uint8)

    @abstractmethod
    def make_mask(self):
        pass

    def get_ifft(self):
        self._fft_image = self._fft_image*self._mask
        self._shifted_fft[self._mask==0] = [0]
        self._process_image = np.abs(np.fft.ifft2(np.fft.fftshift(self._shifted_fft)))

    def set_image(self, ax, fft_mode=False):
        if not fft_mode:
            ax.imshow(self._process_image, cmap='gray')
            ax.set_title(self._title)
        else:
            ax.imshow(self._fft_image, cmap='gray')
            ax.set_title("FFT " + self._title)

        ax.set_xticks([]), ax.set_yticks([]), ax.set_xticklabels([]), ax.set_yticklabels([])

class LowpassFilter(FastFourierTransform):
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
