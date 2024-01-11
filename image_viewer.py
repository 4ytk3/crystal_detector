import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from abc import abstractclassmethod, ABC, ABCMeta, abstractmethod

class Image:
    @abstractmethod
    def set_title(self):
        pass

    @abstractmethod
    def apply_filter(self):
        pass

    @staticmethod
    def read_image(path):
        _rgb_image = ConvertImage.path2image(path)
        _gray_image = ConvertImage.rgb2gray(_rgb_image)
        return _rgb_image, _gray_image

class OriginalImage(Image):
    def __init__(self, title: str, path: str):
        self._title = self.set_title(title)
        self._rgb_image, self._gray_image = self.read_image(path)

    def set_title(self, title: str):
        return "Original " + title

    def apply_filter(self):
        pass

class ConvertImage:
    @staticmethod
    def path2image(image: str) -> np.ndarray:
        _bgr_image = cv2.imread(image)
        _rgb_image = cv2.cvtColor(_bgr_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return _rgb_image

    @staticmethod
    def gray2rgb(image: np.ndarray) -> np.ndarray:
        _rgb_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return _rgb_image

    @staticmethod
    def rgb2gray(image: np.ndarray) -> np.ndarray:
        _gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return _gray_image

class ShowImage: #Static
    @staticmethod
    def set_image(ax, image: Image):
        ax.imshow(image._rgb_image)
        ax.set_title(image._title), ax.set_xticks([]), ax.set_yticks([]), ax.set_xticklabels([]), ax.set_yticklabels([])

    @staticmethod
    def show_image(image: Image):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image)
        plt.show()

    @staticmethod
    def save_image(image: Image, dir='image'):
        name = image._title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.png')
        bgr_image = cv2.cvtColor(image._rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr_image)

    @staticmethod
    def save_fig(image: Image, dir='image'):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image)
        name = image._title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.png')
        fig.savefig(path, dpi=600)


def main(title, path):
    nacl = Image(title, path)
    ShowImage.save_fig(nacl)

if __name__ == '__main__':
    title = "NaCl"
    path = "image/NaCl1_noscale.jpg"
    main(title, path)