import cv2
import numpy as np
from image_processor import Image

class CannyEdgeDetector(Image):
    def __init__(self, image: Image, min_val=100, max_val=400):
        self._title = self.set_title(image._title)
        self._gray_image = self.canny_edge(image._gray_image.copy(), min_val, max_val)

    def set_title(self, title: str):
        return "Canny " + title.replace("Original ", "")

    def canny_edge(self, gray_image: np.ndarray, min_val, max_val):
        return cv2.Canny(gray_image.astype(np.uint8), threshold1=min_val, threshold2=max_val, L2gradient=True)

class PrewittEdgeDetector(Image):
    def __init__(self, image: Image, ksize=3):
        self._title = self.set_title(image._title, ksize)
        self._gray_image = self.prewitt_edge(image._gray_image.copy(), ksize)

    def set_title(self, title: str, ksize: int):
        return f"Prewitt{ksize} " + title.replace("Original ", "")

    def make_kernel(self, ksize):
        if ksize == 3:
            kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        elif ksize == 5:
            kernel_x = np.array([[2,2,2,2,2],[1,1,1,1,1],[0,0,0,0,0],[-1,-1,-1,-1,-1],[-2,-2,-2,-2,-2]])
            kernel_y = np.array([[-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2],[-2,-1,0,1,2]])
        elif ksize == 7:
            kernel_x = np.array([[3,3,3,3,3,3,3],[2,2,2,2,2,2,2],[1,1,1,1,1,1,1],[0,0,0,0,0,0,0],[-1,-1,-1,-1,-1,-1,-1],[-2,-2,-2,-2,-2,-2,-2],[-3,-3,-3,-3,-3,-3,-3]])
            kernel_y = np.array([[-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3]])
        else:
            print(f"ksize={ksize} dosen't support")

        return kernel_x, kernel_y

    def prewitt_edge(self, gray_image: np.ndarray, ksize: int):
        kernel_x, kernel_y = self.make_kernel(ksize)
        prewitt_x = cv2.filter2D(gray_image, -1, kernel_x)
        prewitt_y = cv2.filter2D(gray_image, -1, kernel_y)
        return np.sqrt(prewitt_x**2 + prewitt_y**2).astype(np.float32)

class SobelEdgeDetector(Image):
    def __init__(self, image: Image, dx=1, dy=1, ksize=3, amp=3):
        self._title = self.set_title(image._title, ksize)
        self._gray_image = self.sobel_edge(image._gray_image.copy(), dx, dy, ksize, amp)

    def set_title(self, title: str, ksize: int):
        return f"Sobel{ksize} " + title.replace("Original ", "")

    def sobel_edge(self, gray_image: np.ndarray, dx, dy, ksize, amp):
        return cv2.convertScaleAbs(cv2.Sobel(gray_image.astype(np.uint8), cv2.CV_8U, dx, dy, ksize=ksize)) * amp

class LaplacianEdgeDetector(Image):
    def __init__(self, image: Image, ksize=3, amp=3):
        self._title = self.set_title(image._title, ksize)
        self._gray_image = self.laplacian_edge(image._gray_image.copy(), ksize, amp)

    def set_title(self, title: str, ksize: int):
        return f"Laplacian{ksize} " + title.replace("Original ", "")

    def laplacian_edge(self, gray_image: np.ndarray, ksize, amp):
        return cv2.convertScaleAbs(cv2.Laplacian(gray_image.astype(np.uint8), cv2.CV_8U, ksize=ksize)) * amp