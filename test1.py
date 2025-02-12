from image_processor import Image, Path2Image, ComposeImage, NewImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector, CannyEdgeDetector, PrewittEdgeDetector, SobelEdgeDetector
# from line_detector import HoughTransform, PHoughTransform, LineSegmentDetector

if __name__ == '__main__':
    title = "test"
    path = "nacl.jpg"
    original = Path2Image(title, path)
    highpass = HighpassFilter(original, outer_radius=50)
    Image.show_image(highpass._title, highpass._masked_fft_image)
    peak = PeakFilter(highpass, [[0, 0]])
    Image.show_image(peak._title, peak._spot_image)
    peak_ifft = IFFT(peak)
    Image.show_image(peak_ifft._title, peak_ifft._gray_image)
    bin = OtsuBinarization(peak_ifft)
    Image.show_image(bin._title, bin._gray_image)
    compose = ComposeImage(original, bin)
    Image.show_image(compose._title, compose._rgb_image)