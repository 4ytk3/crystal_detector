from image_processor import Image, Path2Image, ComposeImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector, CannyEdgeDetector, PrewittEdgeDetector, SobelEdgeDetector
from line_detector import HoughTransform, PHoughTransform, LineSegmentDetector

if __name__ == '__main__':
    title = "clear"
    path = "clear/clear.jpg"
    original1 = Path2Image(title, path)
    original2 = Path2Image(title, path)
    bil5 = BilateralFilter(original1, ksize=5)
    low = LowpassFilter(original2, inner_radius=65)
    low_ifft = IFFT(low)
    #can = CannyEdgeDetector(bil5, min_val=250, max_val=260)

    lsd1 = LineSegmentDetector(bil5)
    comp1 = ComposeImage(original1, lsd1)
    #Image.save_image(can._title, can._gray_image, dir='clear/')
    #Image.save_image(bil5._title, bil5._gray_image, dir='clear/')
    #Image.save_image(comp1._title, comp1._rgb_image, dir='clear/')
    #Image.show_image(comp1._title, comp1._rgb_image)

    lap2 = LaplacianEdgeDetector(low_ifft)
    lsd2 = LineSegmentDetector(lap2)
    comp2 = ComposeImage(original2, lsd2)
    #Image.save_image(low_ifft._title, low_ifft._gray_image, dir='clear/')
    Image.save_image(comp2._title, comp2._rgb_image, dir='clear/')
    #Image.show_image(comp2._title, comp2._rgb_image)