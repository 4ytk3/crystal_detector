from image_processor import Image, Path2Image
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter


if __name__ == '__main__':
    title = "test"
    path = "image/test.png"
    original = Path2Image(title, path)
    fft = FFT(original)
    highpass = HighpassFilter(original)
    ifft = IFFT(highpass)
    Image.show_image(ifft._title, ifft._ifft_image)

