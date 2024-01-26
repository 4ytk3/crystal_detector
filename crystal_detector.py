from image_processor import Image, Path2Image
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector
from line_detector import HoughTransform, LineSegmentDetector


if __name__ == '__main__':
    title = "NaCl"
    path = "image/NaCl1_noscale.jpg"
    original = Path2Image(title, path)
    lowpass = LowpassFilter(original)
    Image.show_image(lowpass._title, lowpass._fft_image)
    ifft = IFFT(lowpass)
    Image.show_image(ifft._title, ifft._gray_image)
    edge = LaplacianEdgeDetector(ifft, amp=3)
    Image.show_image(edge._title, edge._gray_image)
    bin = OtsuBinarization(edge)
    Image.show_image(bin._title, bin._gray_image)
    hough = HoughTransform(edge)
    Image.show_image(hough._title, hough._gray_image)
    lsd = LineSegmentDetector(bin)
    Image.show_image(lsd._title, lsd._gray_image)