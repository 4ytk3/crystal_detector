import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from abc import abstractclassmethod, ABC, ABCMeta, abstractmethod

class Image(ABC):
    def __init__(self, image: str, title: str):
        self._image = image
        self._rgb_image = ConvertImage.path2image(self._image)
        self._gray_image = ConvertImage.image2gray(self._rgb_image)
        self._title = title
        self.apply_filter()

    @abstractmethod
    def apply_filter(self, *args):
        pass


class ConvertImage:
    @staticmethod
    def path2image(image: str) -> np.ndarray:
        _bgr_image = cv2.imread(image)
        _rgb_image = cv2.cvtColor(_bgr_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return _rgb_image

    @staticmethod
    def gray2image(image: np.ndarray) -> np.ndarray:
        _rgb_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return _rgb_image

    @staticmethod
    def image2gray(image: np.ndarray) -> np.ndarray:
        _gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return _gray_image


class ShowImage1: #Static
    @staticmethod
    def show_image(image: np.ndarray, title: str):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image, title)
        plt.show()

    @staticmethod
    def save_image(image: np.ndarray, title: str):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image, title)
        name = title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.png')
        if len(image) == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr_image)

    @staticmethod
    def save_fig(image: np.ndarray, title: str):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image, title)
        name = title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.png')
        fig.savefig(path, dpi=600)


class ShowImage(ABC):
    def __init__(self, image, title):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image, title)
        name = title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.png')

    @staticmethod
    def set_image(ax, image: np.ndarray, title: str):
        ax.imshow(image, cmap='gray')
        ax.set_title(title), ax.set_xticks([]), ax.set_yticks([]), ax.set_xticklabels([]), ax.set_yticklabels([])

class SaveImage(ShowImage):
    pass

class SaveFig(ShowImage):
    pass