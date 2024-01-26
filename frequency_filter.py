import cv2
import numpy as np
from image_processor import Image
from abc import ABC, abstractmethod
from scipy.ndimage import maximum_filter

class FFT(Image):
    def __init__(self, image: Image, *args):
        self._title = self.set_title(image._title)
        self._shifted_fft = self.get_fft(image._gray_image)
        self._fft_image = self.get_spectrum(self._shifted_fft)
        self._mask = self.make_mask(self._fft_image, *args)
        self._masked_shifted_fft, self._masked_fft_image = self.apply_mask(self._shifted_fft, self._fft_image, self._mask)

    def set_title(self, title: str) -> str:
        return "FFT " + title.replace("Original ", "")

    def get_fft(self, gray_image):
        return np.fft.fftshift(np.fft.fft2(gray_image))

    def get_spectrum(self, shifted_fft):
        return 20*np.log(np.abs(shifted_fft))

    def make_mask(self, fft_image):
        height, width = fft_image.shape[0], fft_image.shape[1]
        mask = np.ones([height, width], dtype=np.uint8)
        return mask

    def apply_mask(self, shifted_fft, fft_image, mask):
        masked_shifted_fft = shifted_fft[mask==0] = 0
        masked_fft_image = fft_image*mask
        return masked_shifted_fft, masked_fft_image

class IFFT(Image):
    def __init__(self, image: FFT, *args):
        self._title = self.set_title(image._title)
        self._gray_image = self.get_ifft(image._shifted_fft)

    def set_title(self, title: str) -> str:
        return "IFFT " + title.replace("FFT", "")

    def get_ifft(self, shifted_fft):
        return np.abs(np.fft.ifft2(np.fft.fftshift(shifted_fft)))


class LowpassFilter(FFT):
    def __init__(self, image: Image, inner_radius=90):
        super().__init__(image, inner_radius)

    def set_title(self, title: str):
        return "Lowpass " + title.replace("Original ", "")

    def make_mask(self, image, inner_radius):
        height, width = image.shape[0], image.shape[1]
        mask = np.zeros([height, width], dtype=np.uint8)
        cv2.circle(mask, center=(width//2, height//2), radius=inner_radius, color=255, thickness=-1)
        return mask

class HighpassFilter(FFT):
    def __init__(self, fft_image: Image, outer_radius=80):
        super().__init__(fft_image, outer_radius)

    def set_title(self, title: str):
        return "Highpass " + title.replace("Original ", "")

    def make_mask(self, shifted_fft, outer_radius):
        height, width = shifted_fft.shape[0], shifted_fft.shape[1]
        mask = np.ones([height, width], dtype=np.uint8)
        cv2.circle(mask, center=(width//2, height//2), radius=outer_radius, color=0, thickness=-1)
        return mask

class BandpassFilter(FFT):
    def __init__(self, fft_image: Image, outer_radius=100, inner_radius=50):
        super().__init__(fft_image, outer_radius, inner_radius)

    def set_title(self, title: str):
        return "Bandpass " + title.replace("Original ", "")

    def make_mask(self, shifted_fft, outer_radius, inner_radius):
        height, width = shifted_fft.shape[0], shifted_fft.shape[1]
        mask = np.zeros([height, width], dtype=np.uint8)
        cv2.circle(mask, center=(width//2, height//2), radius=outer_radius, color=1, thickness=-1)
        cv2.circle(mask, center=(width//2, height//2), radius=inner_radius, color=0, thickness=-1)
        return mask

class PeakFilter(FFT):
    def __init__(self, image: FFT):
        self._title = self.set_title(image._title)
        self._peak_image = self.detect_peaks(image._fft_image)
        self._spot_image = self.peak2spot(self._peak_image)
        self._fft_image = image._fft_image*self._spot_image
        self._shifted_fft = image._shifted_fft
        self._shifted_fft[self._spot_image==0] = 0

    def set_title(self, title: str):
        return "Peak " + title.replace("Original ", "")

    def detect_peaks(self, fft_image: np.ndarray, filter_size=5, order=0.8):
        peak_image = fft_image
        local_max = maximum_filter(peak_image, footprint=np.ones((filter_size, filter_size)), mode='constant')
        peak_image[local_max!=peak_image] = [0]
        peak_image[peak_image<=peak_image.max()*order] = [0]
        peak_image[peak_image!=0] = [1]
        return peak_image

    def peak2spot(self, peak_image: np.ndarray):
        spot_image = peak_image.copy()
        indices = np.dstack(np.where(spot_image == 1))
        for index in indices[0]:
            cv2.circle(spot_image, center=(index[1], index[0]), radius=5, color=1, thickness=-1)
        return spot_image