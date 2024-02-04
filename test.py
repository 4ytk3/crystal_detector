from image_processor import Image, Path2Image, ComposeImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector, CannyEdgeDetector, PrewittEdgeDetector, SobelEdgeDetector
from line_detector import HoughTransform, PHoughTransform, LineSegmentDetector
import glob
import time
import numpy as np

if __name__ == '__main__':
    start = time.time()
    images = glob.glob("./nacl./*.jpg")
    n=0
    for i in images:
        title = "NaCl"
        path = i
        original = Path2Image(title, path)
        bandpass = BandpassFilter(original, outer_radius=100, inner_radius=50)
        peak = PeakFilter(bandpass)
        #Image.show_image(peak._title, peak._fft_image)
        Image.save_image(f"{n+1}_peak", peak._fft_image, dir='ditected/')
        peak_ifft = IFFT(peak)
        #Image.show_image(peak_ifft._title, peak_ifft._gray_image)
        bin = OtsuBinarization(peak_ifft)
        #Image.show_image(bin._title, bin._gray_image)
        compose = ComposeImage(original, bin)
        blue_pixel = Image.pixel_counter(compose._title, compose._rgb_image)
        save_title = f'{n+1}_{blue_pixel}'
        #Image.show_image(save_title, compose._rgb_image)
        Image.save_image(save_title, compose._rgb_image, dir='ditected/')
        n+=1
    time.sleep(0.1)
    end = time.time() - start
    print(f"run-time is {end}")