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
    def gray2rgb(image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_GRAY2RGB)
        #rgb_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return rgb_image

    @staticmethod
    def rgb2gray(image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        #gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return gray_image

    @staticmethod
    def set_image(ax, title: str, image: np.ndarray):
        ax.imshow(image, cmap='gray')
        ax.set_title(title), ax.set_xticks([]), ax.set_yticks([]), ax.set_xticklabels([]), ax.set_yticklabels([])

    @staticmethod
    def show_image(title: str, image: np.ndarray):
        fig, ax = plt.subplots()
        Image.set_image(ax, title, image)
        plt.show()

    @staticmethod
    def save_image(title: str, image: np.ndarray, dir='image', fig_mode=False):
        name = title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.jpg')
        if fig_mode:
            fig, ax = plt.subplots()
            Image.set_image(ax, title, image)
            fig.savefig(path, dpi=600)
        elif len(image.shape) == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr_image)
        else:
            #image = np.clip(image * 255, 0, 255)
            cv2.imwrite(path, image.astype(np.float32))
            print("saved")

    @staticmethod
    def pixel_counter(title: str, image: np.ndarray, color=[0,0,255]):
        counter = 0
        color = np.array(color)
        height, width = image.shape[0], image.shape[1]
        for i in range(height):
            for j in range(width):
                pixel = image[i, j]
                if (pixel == color).all():
                    counter += 1
        print(f"{title}'s number of {color} pixel is {counter}")
        return counter

    @staticmethod
    def calc_psnr(title: str, original: np.ndarray, image: np.ndarray):
        psnr = cv2.PSNR(original, image.astype(np.float32))
        print(f"{title}'s PSNR is {psnr}")

    @staticmethod
    def calc_ssim(title: str, original: np.ndarray, image: np.ndarray):
        ssim, _ = cv2.quality.QualitySSIM_compute(original, image)
        print(f"{title}'s SSIM is {ssim[0]}")

class ComposeImage(Image):
    def __init__(self, original_image: Image, bin_image: Image):
        # rgb_image = original_image._rgb_image.astype(np.uint8).copy()
        # gray_image = bin_image._gray_image.astype(np.uint8).copy()
        self._title = self.set_title(bin_image._title)
        self._rgb_image = self.compose_image(original_image._rgb_image, bin_image._gray_image)

    def set_title(self, title: str) -> str:
        return "Composed " + title

    def compose_image(self, original_image: np.ndarray, bin_image: np.ndarray):
        rgb_original = original_image.copy()
        rgb_bin = self.gray2rgb(bin_image.copy())
        rgb_original[(rgb_bin==(255,255,255)).all(axis=-1)]=(0, 0, 255)
        return rgb_original


class Path2Image(Image):
    def __init__(self, title: str, path: str):
        self._title = self.set_title(title)
        self._rgb_image = self.path2rgb(path)
        self._gray_image = self.rgb2gray(self._rgb_image)

    def set_title(self, title: str) -> str:
        return "Original " + title

    @staticmethod
    def path2rgb(path: str) -> np.ndarray:
        bgr_image = cv2.imread(path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #rgb_image = cv2.cvtColor(bgr_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rgb_image