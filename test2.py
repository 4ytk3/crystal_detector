from image_processor import Image, Path2Image, ComposeImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector, CannyEdgeDetector, PrewittEdgeDetector, SobelEdgeDetector
from line_detector import HoughTransform, PHoughTransform, LineSegmentDetector

if __name__ == '__main__':
    title = "unclear"
    path = "unclear/unclear.png"
    original = Path2Image(title, path)

    bil5 = BilateralFilter(original, ksize=5)
    lap_bil5 = LaplacianEdgeDetector(bil5, ksize=5)
    Image.save_image(lap_bil5._title, lap_bil5._gray_image, dir='unclear/')
    lsd_bil5 = LineSegmentDetector(bil5)
    lsd_can_bil5 = LineSegmentDetector(lap_bil5)
    comp_lsd_bil5 = ComposeImage(original, lsd_bil5)
    comp_lsd_can_bil5 = ComposeImage(original, lsd_can_bil5)
    Image.save_image(comp_lsd_bil5._title, comp_lsd_bil5._rgb_image, dir='unclear/')
    Image.save_image(comp_lsd_can_bil5._title, comp_lsd_can_bil5._rgb_image, dir='unclear/')
    #Image.show_image(lap_bil5._title, lap_bil5._gray_image)
    #Image.show_image(comp_lsd_can_bil5._title, comp_lsd_can_bil5._rgb_image)

    low = LowpassFilter(original, inner_radius=65)
    low_ifft = IFFT(low)
    lap_low = LaplacianEdgeDetector(low_ifft, ksize=5)
    Image.save_image(lap_low._title, lap_low._gray_image, dir='unclear/')
    lsd_low = LineSegmentDetector(low_ifft)
    lsd_lap_low = LineSegmentDetector(lap_low)
    comp_lsd_low = ComposeImage(original, lsd_low)
    comp_lsd_lap_low = ComposeImage(original, lsd_lap_low)
    Image.save_image(comp_lsd_low._title, comp_lsd_low._rgb_image, dir='unclear/')
    Image.save_image(comp_lsd_lap_low._title, comp_lsd_lap_low._rgb_image, dir='unclear/')
    #Image.show_image(comp2._title, comp2._rgb_image)