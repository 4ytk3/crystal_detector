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

    highpass1 = HighpassFilter(original, outer_radius=50)
    highpass2 = HighpassFilter(original, outer_radius=50)
    # Image.show_image(highpass._title, highpass._masked_fft_image)
    center = 540
    Y = 2
    X = 60
    indices1 = [[center-X, center-Y], [center+X, center+Y], [center, center]]
    indices2 = [[center+X, center+Y], [center-X, center-Y], [center, center]]
    peak1 = PeakFilter(highpass1, indices1)
    peak2 = PeakFilter(highpass2, [[0, 0]])
    Image.show_image(peak1._title, peak1._spot_image)
    Image.show_image(peak2._title, peak2._spot_image)
    peak1_ifft = IFFT(peak1)
    peak2_ifft = IFFT(peak2)
    Image.show_image(peak1_ifft._title, peak1_ifft._gray_image)
    Image.show_image(peak2_ifft._title, peak2_ifft._gray_image)
    bin1 = OtsuBinarization(peak1_ifft)
    bin2 = OtsuBinarization(peak2_ifft)
    Image.show_image(bin1._title, bin1._gray_image)
    Image.show_image(bin2._title, bin2._gray_image)
    bin = NewImage("nacl", bin1._gray_image - bin2._gray_image)
    Image.show_image(bin._title, bin._gray_image)
    compose = ComposeImage(original, bin)
    # compose1 = ComposeImage(original, bin1)
    # compose2 = ComposeImage(original, bin2)
    Image.show_image(compose._title, compose._rgb_image)
    # Image.show_image(compose1._title, compose1._rgb_image)
    # Image.show_image(compose2._title, compose2._rgb_image)
