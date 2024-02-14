from image_processor import Image, Path2Image, ComposeImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector, CannyEdgeDetector, PrewittEdgeDetector, SobelEdgeDetector
from line_detector import HoughTransform, PHoughTransform, LineSegmentDetector

if __name__ == '__main__':
    title = "clear"
    path = "clear/clear.png"
    original = Path2Image(title, path)

    # bil5 = BilateralFilter(original, ksize=5)
    # can_bil5 = CannyEdgeDetector(bil5, min_val=100, max_val=200)
    # #lap_bil5 = LaplacianEdgeDetector(bil5, ksize=5)
    # #Image.save_image(can_bil5._title, can_bil5._gray_image, dir='unclear/')
    # #lsd_bil5 = LineSegmentDetector(bil5)
    # lsd_can_bil5 = LineSegmentDetector(can_bil5)
    # #comp_lsd_bil5 = ComposeImage(original, lsd_bil5)
    # comp_lsd_can_bil5 = ComposeImage(original, lsd_can_bil5)
    # #Image.save_image(comp_lsd_bil5._title, comp_lsd_bil5._rgb_image, dir='unclear/')
    # Image.save_image(comp_lsd_can_bil5._title, comp_lsd_can_bil5._rgb_image, dir='unclear/')
    # Image.show_image(comp_lsd_can_bil5._title, comp_lsd_can_bil5._rgb_image)

    # low = LowpassFilter(original, inner_radius=65)
    # low_ifft = IFFT(low)
    # can_low = CannyEdgeDetector(low_ifft, min_val=120, max_val=250)
    # #Image.save_image(lap_low._title, lap_low._gray_image, dir='unclear/')
    # #lsd_low = LineSegmentDetector(low_ifft)
    # lsd_can_low = LineSegmentDetector(can_low)
    # #comp_lsd_low = ComposeImage(original, lsd_low)
    # comp_lsd_can_low = ComposeImage(original, lsd_can_low)
    # Image.save_image(can_low._title, can_low._gray_image, dir='unclear/')
    # Image.save_image(comp_lsd_can_low._title, comp_lsd_can_low._rgb_image, dir='unclear/')
    # Image.show_image(comp_lsd_can_low._title, comp_lsd_can_low._rgb_image)

    bandpass = BandpassFilter(original, outer_radius=100, inner_radius=50)
    Image.show_image(bandpass._title, bandpass._masked_fft_image)
    peak = PeakFilter(bandpass)
    Image.show_image(peak._title, peak._fft_image)
    peak_ifft = IFFT(peak)
    Image.show_image(peak_ifft._title, peak_ifft._gray_image)
    bin = OtsuBinarization(peak_ifft)
    Image.show_image(bin._title, bin._gray_image)
    compose = ComposeImage(original, bin)
    Image.show_image(compose._title, compose._rgb_image)