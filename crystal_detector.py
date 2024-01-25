from image_processor import Image, Path2Image
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from edge_detector import PrewittEdgeDetector


if __name__ == '__main__':
    title = "test"
    path = "image/NaCl1_noscale.jpg"
    original = Path2Image(title, path)
    prewitt = PrewittEdgeDetector(original, 3)
    Image.show_image(original._title, original._gray_image)
    Image.show_image(prewitt._title, prewitt._gray_image)

