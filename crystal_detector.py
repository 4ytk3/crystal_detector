from image_processor import Image, OriginalImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter


if __name__ == '__main__':
    title = "test"
    path = "image/NaCl1_noscale.jpg"
    original = OriginalImage(title, path)
    fft = FFT(original)
    ifft = IFFT(fft)
    Image.show_image(fft._title, fft._fft_image)
    Image.show_image(ifft._title, ifft._ifft_image)

