from __future__ import annotations
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod

class Image(ABC): # Interface for Image
    @abstractmethod
    def set_title(self):
        raise NotImplementedError()

    @staticmethod
    def path2rgb(path: str) -> np.ndarray:
        bgr_image = cv2.imread(path)
        rgb_image = cv2.cvtColor(bgr_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rgb_image

    @staticmethod
    def gray2rgb(image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return rgb_image

    @staticmethod
    def rgb2gray(image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return gray_image

class OriginalImage(Image):
    def __init__(self, title: str, path: str):
        self._title = self.set_title(title)
        self._rgb_image = self.path2rgb(path)
        self._gray_image = self.rgb2gray(self._rgb_image)

    def set_title(self, title: str) -> str:
        return "Original " + title

class ImageProcessor:
    @staticmethod
    def set_image(ax, image: Image):
        ax.imshow(image._rgb_image)
        ax.set_title(image._title), ax.set_xticks([]), ax.set_yticks([]), ax.set_xticklabels([]), ax.set_yticklabels([])

    @staticmethod
    def show_image(image: Image):
        fig, ax = plt.subplots()
        ImageProcessor.set_image(ax, image)
        plt.show()

    @staticmethod
    def save_image(image: Image, dir='image', fig_mode=False):
        name = image._title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.png')
        if fig_mode:
            fig, ax = plt.subplots()
            ImageProcessor.set_image(ax, image)
            fig.savefig(path, dpi=600)
        else:
            bgr_image = cv2.cvtColor(image._rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr_image)

if __name__ == '__main__':
    title = "NaCl"
    path = "image/NaCl1_noscale.jpg"
    nacl = OriginalImage(title, path)
    ImageProcessor.show_image(nacl)