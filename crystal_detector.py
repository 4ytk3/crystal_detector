import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from abc import abstractclassmethod, ABC, ABCMeta, abstractmethod

from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max

import math
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pylsd import lsd

class Image(ABC):
    def __init__(self, image: str, title: str):
        self._image = LoadImage(image)
        self._title = title
        self.apply_filter()

    @abstractmethod
    def apply_filter(self, *args):
        pass

class Main:
    def __init__(self, filter: object = Image):
        ShowImage.show_image()

class LoadImage: # Static
    def __init__(self, image: str):
        if type(image) is str:
            self._rgb_image = LoadImage.path2image(image)
            self._gray_image = LoadImage.image2gray(self._rgb_image)

        elif len(image.shape) == 2:
            self._gray_image = image
            self._rgb_image = LoadImage.gray2image(self._gray_image)

        elif len(image.shape) == 4:
            self._rgb_image = image
            self._gray_image = LoadImage.image2gray(image)

        else:
            print(f"This is not image")
            sys.exit()

    @staticmethod
    def path2image(image: str) -> np.ndarray:
        try:
            _rgb_image = cv2.imread(image)
            _rgb_image = cv2.cvtColor(_rgb_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        except:
            print(f"Can't load image: {image}")
            sys.exit()
        return _rgb_image

    @staticmethod
    def gray2image(image: np.ndarray) -> np.ndarray:
        _gray_image = image
        _rgb_image = cv2.cvtColor(_gray_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return _rgb_image

    @staticmethod
    def image2gray(image: np.ndarray) -> np.ndarray:
        _gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return _gray_image

class ShowImage: #Static
    @staticmethod
    def set_image(ax, image: np.ndarray, title: str):
        ax.imshow(image, cmap='gray')
        ax.set_title(title), ax.set_xticks([]), ax.set_yticks([]), ax.set_xticklabels([]), ax.set_yticklabels([])

    @staticmethod
    def show_image(image: np.ndarray, title: str):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image, title)
        plt.show()

    @staticmethod
    def save_image(image: np.ndarray, title: str, fig_mode: bool=True, dir: str='filtered_images'):
        fig, ax = plt.subplots()
        ShowImage.set_image(ax, image, title)
        name = title.lower().replace(' ', '_')
        path = os.path.join(dir, f'{name}.png')
        if not fig_mode:
            if len(image) == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr_image)
        else:
            fig.savefig(path, dpi=600)

class FastFourierTransform(Image):
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
    def __init__(self, title, image, outer_radius=40, inner_radius=32):
        self._outer_radius, self._inner_radius = outer_radius, inner_radius
        super().__init__(title, image)
        self._title = "Peak " + self._title

    def make_mask(self):
        self._circle_mask = np.zeros([self._height, self._width], dtype=np.uint8)
        cv2.circle(self._circle_mask, center=(self._center_x, self._center_y), radius=self._outer_radius, color=1, thickness=-1)
        cv2.circle(self._circle_mask, center=(self._center_x, self._center_y), radius=self._inner_radius, color=0, thickness=-1)
        self._bandpass_image = self._fft_image*self._circle_mask
        self.view_image(image=self._bandpass_image)
        self.detect_peaks()
        self.indices = np.dstack(np.where(self._bandpass_image == 1))
        for index in self.indices[0]:
            cv2.circle(self._bandpass_image, center=(index[0], index[1]), radius=5, color=1, thickness=-1)
        self.view_image(image=self._bandpass_image)
        self._mask = self._bandpass_image

    def detect_peaks(self, filter_size=5, order=0.5):
        self.__local_max = maximum_filter(self._bandpass_image, footprint=np.ones((filter_size, filter_size)), mode='constant')
        self.view_image(image=self.__local_max)
        self._bandpass_image[self.__local_max!=self._bandpass_image] = [0]
        self._bandpass_image[self._bandpass_image.max()!=self._bandpass_image] = [0]
        self._bandpass_image[self._bandpass_image!=0] = [1]
        self.view_image(image=self._bandpass_image)

class AverageFilter(Image):
    def __init__(self, image: str, title: str):
        super().__init__(image, title)
        self._title = "Average " + self._title

    def apply_filter(self, ksize: int=3):
        self._process_image = cv2.blur(src=self._gray_image, ksize=(ksize,ksize))


class GaussianFilter(Image):
    def __init__(self, title, image, ksize=3, sigmaX=3):
        super().__init__(title, image)
        self._title = "Gaussian " + self._title
        self._process_image = cv2.GaussianBlur(src=self._gray_image, ksize=(ksize,ksize), sigmaX=sigmaX)

class MedianFilter(Image):
    def __init__(self, title, image, ksize=3):
        super().__init__(title, image)
        self._title = "Median " + self._title
        self._process_image = cv2.medianBlur(src=np.float32(self._gray_image), ksize=ksize)

class BilateralFilter(Image):
    def __init__(self, title, image, ksize=3, sigmaColor=10, sigmaSpace=10):
        super().__init__(title, image)
        self._title = "Bilateral " + self._title
        self._process_image = cv2.bilateralFilter(src=np.float32(self._gray_image), d=ksize, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

class Binarization(Image):
    def __init__(self, title, image):
        super().__init__(title, image)
        self._title = "Binarization " + self._title
        self.otsu_bin()

    def otsu_bin(self, threshold=128):
        ret, self._process_image = cv2.threshold(self._gray_image, threshold, 255, cv2.THRESH_OTSU)

    def adapt_bin(self, blockSize=5):
        self._process_image = cv2.adaptiveThreshold(self._gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=blockSize, C=0)

class CannyEdgeDetector(Image):
    def __init__(self, title, image, min_val=70, max_val=150):
        super().__init__(title, image)
        self._title = "Canny " + self._title
        self.canny_edge(min_val, max_val)

    def canny_edge(self, min_val, max_val):
        self._process_image = cv2.Canny(self._gray_image.astype(np.uint8), threshold1=min_val, threshold2=max_val, L2gradient=True)

class SobelEdgeDetector(Image):
    def __init__(self, title, image, dx=1, dy=1, ksize=3):
        super().__init__(title, image)
        self._title = "Sobel " + self._title
        self.sobel_edge(dx, dy, ksize)

    def sobel_edge(self, dx, dy, ksize):
        self._process_image = cv2.Sobel(self._gray_image.astype(np.uint8), cv2.CV_8U, dx, dy, ksize=ksize)
        self._process_image = cv2.convertScaleAbs(self._process_image)
        self._process_image = self._process_image * 3

class LaplacianEdgeDetector(Image):
    def __init__(self, title, image):
        super().__init__(title, image)
        self._title = "Laplacian " + self._title
        self.laplacian_edge()

    def laplacian_edge(self):
        self._process_image = cv2.Laplacian(self._gray_image.astype(np.uint8), cv2.CV_8U)
        self._process_image = cv2.convertScaleAbs(self._process_image)
        self._process_image = self._process_image * 10

class HoughTransform(Image):
    def __init__(self, title, image):
        super().__init__(title, image)
        self._title = "Hough " + self._title
        self.draw_lines()

    def draw_lines(self):
        _lines = cv2.HoughLines(self._gray_image.astype(np.uint8), 1, np.pi/360, 200)
        if _lines is not None:
            for line in _lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(self._process_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        else:
            print("Can't draw ht lines")

class PHoughTransform(Image):
    def __init__(self, title, image, threshold=500, minLineLength=30, maxLineGap=50):
        super().__init__(title, image)
        self._title = "PHough " + self._title
        self.draw_lines(threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    def draw_lines(self, threshold, minLineLength, maxLineGap):
        __lines = cv2.HoughLinesP(self._gray_image.astype(np.uint8), 10, np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
        if __lines is not None:
            degs = []
            for line in __lines:
                x1,y1,x2,y2 = line[0]
                rad = math.atan2(x2-x1, y2-y1)
                deg = rad*(180/np.pi)
                deg = int(Decimal(deg).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP)) #10の位に丸める
                degs.append(deg)
            mode = statistics.mode(degs)
            for line in __lines:
                x1,y1,x2,y2 = line[0]
                rad = math.atan2(x2-x1, y2-y1)
                deg = rad*(180/np.pi)
                deg = int(Decimal(deg).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP))
                cv2.line(self._process_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # if deg <= mode+5 and deg >= mode-5:
                #     cv2.line(self._image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        else:
            print("Can't draw pht lines")

class LineSegmentDetector(Image):
    def __init__(self, title, image):
        super().__init__(title, image)
        self._title = "LSD " + self._title
        self.detect_lines()

    def detect_lines(self):
        lines = lsd(self._gray_image)
        if lines is not None:
            degs = []
            line_infos = []
            for line in lines:
                x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                rad = math.atan2(x2-x1, y2-y1)
                deg = rad*(180/np.pi)
                deg = int(Decimal(deg).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
                if deg <= 0:
                    deg += 180
                degs.append(deg)
                tmp_list = [x1, y1, x2, y2, deg]
                line_infos.append(tmp_list)
            mode = statistics.mode(degs)
            mode_deg_lines = []
            for line_info in line_infos:
                x1, y1, x2, y2, deg = line_info
                # if deg <= mode+20 and deg >= mode-20:
                #     cv2.line(self._process_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                #     mode_deg_line = [x1, y1, x2, y2, deg]
                #     mode_deg_lines.append(mode_deg_line)
                cv2.line(self._process_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        else:
            print("Can't draw lsd lines")

class FastLineDetector(Image): #ToDo
    def __init__(self):
        super().__init__(title, image)
        self._title = "LSD " + self._title
        self.detect_lines()

    def testFastLineDetector(fileImage):
        colorimg = cv2.imread(fileImage, cv2.IMREAD_COLOR)
        if colorimg is None:
            return -1
        image = cv2.cvtColor(colorimg.copy(), cv2.COLOR_BGR2GRAY)

        # FLDインスタンス生成
        length_threshold = 4 # 10
        distance_threshold = 1.41421356
        canny_th1 = 50.0
        canny_th2 = 50.0
        canny_aperture_size = 3
        do_merge = False

        # 高速ライン検出器生成
        fld = cv2.ximgproc.createFastLineDetector(length_threshold,distance_threshold,
                        canny_th1,canny_th2,canny_aperture_size,do_merge)
        #fld = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD) # LSD

        # ライン取得
        lines = fld.detect(image)
        #lines, width, prec, nfa = fld.detect(image) # LSD

        # 検出線表示
        drawnLines = np.zeros((image.shape[0],image.shape[1],3), np.uint8)
        fld.drawSegments(drawnLines, lines)
        cv2.imshow("Fast Line Detector(LINE)", drawnLines)

        # 検出線と処理画像の合成表示
        fld.drawSegments(colorimg, lines)
        cv2.imshow("Fast Line Detector", colorimg)
        cv2.waitKey()
        return 0

class DrawEdgeLine(Image):
    def __init__(self, title, image):
        super().__init__(title, image)
        self._title = "DrawEdgeLine" + self._title
        self.draw_lines()

    def draw_lines(self, ):
        w, h = self._image.shape[:2]
        source_color = (0, 0, 0)
        target_color = (255, 255, 255)
        change_color = (255, 0, 0)

        source_list = [(i, j) for j in range(h) for i in range(w) if tuple(self._image[i, j]) == source_color for l in range(j-1, j+2) for k in range(i-1, i+2) if 0 <= k < w and 0 <= l < h if tuple(self._image[k, l]) == target_color]
        for i, j in source_list:
            self._image[i, j] = change_color


class CrystalDetector:
    def __init__(self, image, title):
        nacl = Image(title, image)
        #nacl1=69,74,66 nacl2=86
        # fft = image_filter.FastFourierTransform(title, image)
        #fft.view_image(fft_mode=True)
        #bandpass.view_image(fft_mode=True)
        peak = PeakFilter(title, image)
        peak.view_image(fft_mode=True)
        peak.view_image()

        # doubleband = image_filter.DoubleBandFilter(title, image, angle=42, size=8, outer_radius=40, inner_radius=32)
        binary = Binarization(peak.title, peak.process_image.astype(np.uint8))
        binary.view_image()
        binary._process_image = cv2.cvtColor(binary._process_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        nacl._process_image[(binary._process_image==(255, 255, 255)).all(axis=-1)]=(0, 0, 200)
        nacl.view_image(save_mode=True, fig_mode=False)

if __name__ == '__main__':
    title = 'NaCl'
    image = 'NaClcrystal_image.jpg'

    nacl = Image(title, image)
=======
ｃ
>>>>>>> origin/main
