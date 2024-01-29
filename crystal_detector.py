from image_processor import Image, Path2Image, ComposeImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector
from line_detector import HoughTransform, LineSegmentDetector


if __name__ == '__main__':
    title = "NaCl"
    path = "image/10033_nacl 001.jpg"
    original = Path2Image(title, path)
    bandpass = BandpassFilter(original)
    Image.show_image(bandpass._title, bandpass._masked_fft_image)
    peak = PeakFilter(bandpass)
    Image.save_image(peak._title, peak._peak_image)
    Image.show_image(peak._title, peak._spot_image)
    ifft = IFFT(peak)
    Image.show_image(ifft._title, ifft._gray_image)
    bin = OtsuBinarization(ifft)
    Image.show_image(bin._title, bin._gray_image)
    compose = ComposeImage(original, bin)
    Image.show_image(compose._title, compose._rgb_image)