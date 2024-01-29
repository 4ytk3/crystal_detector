from image_processor import Image, Path2Image, ComposeImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector, CannyEdgeDetector, PrewittEdgeDetector, SobelEdgeDetector
from line_detector import HoughTransform, PHoughTransform, LineSegmentDetector
import cv2

if __name__ == '__main__':
    # title = "unclear"
    # path = "image/unclear.jpg"
    title = "clear"
    path = "image/clear.jpg"
    original = Path2Image(title, path)


    # Noise Filter
    ave3 = AverageFilter(original, ksize=3)
    ave5 = AverageFilter(original, ksize=5)
    med3 = MedianFilter(original, ksize=3)
    med5 = MedianFilter(original, ksize=5)
    gau3 = GaussianFilter(original, ksize=3)
    gau5 = GaussianFilter(original, ksize=5)
    bil3 = BilateralFilter(original, ksize=3)
    bil5 = BilateralFilter(original, ksize=5)
    low = LowpassFilter(original, inner_radius=65)
    low_ifft = IFFT(low)

    # Edge Filter
    can_ave3 = CannyEdgeDetector(ave3)
    can_ave5 = CannyEdgeDetector(ave5)
    high_ave3 = HighpassFilter(ave3)
    high_ave5 = HighpassFilter(ave5)
    pre3_ave3 = PrewittEdgeDetector(ave3, ksize=3)
    pre3_ave5 = PrewittEdgeDetector(ave5, ksize=3)
    pre5_ave3 = PrewittEdgeDetector(ave3, ksize=5)
    pre5_ave5 = PrewittEdgeDetector(ave5, ksize=5)
    sob3_ave3 = SobelEdgeDetector(ave3, ksize=3)
    sob3_ave5 = SobelEdgeDetector(ave5, ksize=3)
    sob5_ave3 = SobelEdgeDetector(ave3, ksize=5)
    sob5_ave5 = SobelEdgeDetector(ave5, ksize=5)
    lap3_ave3 = LaplacianEdgeDetector(ave3, ksize=3)
    lap3_ave5 = LaplacianEdgeDetector(ave5, ksize=3)
    lap5_ave3 = LaplacianEdgeDetector(ave3, ksize=5)
    lap5_ave5 = LaplacianEdgeDetector(ave5, ksize=5)

    can_med3 = CannyEdgeDetector(med3)
    can_med5 = CannyEdgeDetector(med5)
    high_med3 = HighpassFilter(med3)
    high_med5 = HighpassFilter(med5)
    pre3_med3 = PrewittEdgeDetector(med3, ksize=3)
    pre3_med5 = PrewittEdgeDetector(med5, ksize=3)
    pre5_med3 = PrewittEdgeDetector(med3, ksize=5)
    pre5_med5 = PrewittEdgeDetector(med5, ksize=5)
    sob3_med3 = SobelEdgeDetector(med3, ksize=3)
    sob3_med5 = SobelEdgeDetector(med5, ksize=3)
    sob5_med3 = SobelEdgeDetector(med3, ksize=5)
    sob5_med5 = SobelEdgeDetector(med5, ksize=5)
    lap3_med3 = LaplacianEdgeDetector(med3, ksize=3)
    lap3_med5 = LaplacianEdgeDetector(med5, ksize=3)
    lap5_med3 = LaplacianEdgeDetector(med3, ksize=5)
    lap5_med5 = LaplacianEdgeDetector(med5, ksize=5)

    can_gau3 = CannyEdgeDetector(gau3)
    can_gau5 = CannyEdgeDetector(gau5)
    high_gau3 = HighpassFilter(gau3)
    high_gau5 = HighpassFilter(gau5)
    pre3_gau3 = PrewittEdgeDetector(gau3, ksize=3)
    pre3_gau5 = PrewittEdgeDetector(gau5, ksize=3)
    pre5_gau3 = PrewittEdgeDetector(gau3, ksize=5)
    pre5_gau5 = PrewittEdgeDetector(gau5, ksize=5)
    sob3_gau3 = SobelEdgeDetector(gau3, ksize=3)
    sob3_gau5 = SobelEdgeDetector(gau5, ksize=3)
    sob5_gau3 = SobelEdgeDetector(gau3, ksize=5)
    sob5_gau5 = SobelEdgeDetector(gau5, ksize=5)
    lap3_gau3 = LaplacianEdgeDetector(gau3, ksize=3)
    lap3_gau5 = LaplacianEdgeDetector(gau5, ksize=3)
    lap5_gau3 = LaplacianEdgeDetector(gau3, ksize=5)
    lap5_gau5 = LaplacianEdgeDetector(gau5, ksize=5)

    can_bil3 = CannyEdgeDetector(bil3)
    can_bil5 = CannyEdgeDetector(bil5)
    high_bil3 = HighpassFilter(bil3)
    high_bil5 = HighpassFilter(bil5)
    pre3_bil3 = PrewittEdgeDetector(bil3, ksize=3)
    pre3_bil5 = PrewittEdgeDetector(bil5, ksize=3)
    pre5_bil3 = PrewittEdgeDetector(bil3, ksize=5)
    pre5_bil5 = PrewittEdgeDetector(bil5, ksize=5)
    sob3_bil3 = SobelEdgeDetector(bil3, ksize=3)
    sob3_bil5 = SobelEdgeDetector(bil5, ksize=3)
    sob5_bil3 = SobelEdgeDetector(bil3, ksize=5)
    sob5_bil5 = SobelEdgeDetector(bil5, ksize=5)
    lap3_bil3 = LaplacianEdgeDetector(bil3, ksize=3)
    lap3_bil5 = LaplacianEdgeDetector(bil5, ksize=3)
    lap5_bil3 = LaplacianEdgeDetector(bil3, ksize=5)
    lap5_bil5 = LaplacianEdgeDetector(bil5, ksize=5)

    can_low_ifft = CannyEdgeDetector(low_ifft)
    high_low_ifft = HighpassFilter(low_ifft)
    pre3_low_ifft = PrewittEdgeDetector(low_ifft, ksize=3)
    pre5_low_ifft = PrewittEdgeDetector(low_ifft, ksize=5)
    sob3_low_ifft = SobelEdgeDetector(low_ifft, ksize=3)
    sob5_low_ifft = SobelEdgeDetector(low_ifft, ksize=5)
    lap3_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=3)
    lap5_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=5)

    # Line Detection
    can_ave3 = LineSegmentDetector(can_ave3)
    can_ave5 = LineSegmentDetector(can_ave5)
    high_ave3 = LineSegmentDetector(high_ave3)
    high_ave5 = LineSegmentDetector(high_ave5)
    pre3_ave3 = LineSegmentDetector(pre3_ave3)
    pre3_ave5 = LineSegmentDetector(pre3_ave5)
    pre5_ave3 = LineSegmentDetector(pre5_ave3)
    pre5_ave5 = LineSegmentDetector(pre5_ave5)
    sob3_ave3 = LineSegmentDetector(sob3_ave3)
    sob3_ave5 = LineSegmentDetector(sob3_ave5)
    sob5_ave3 = LineSegmentDetector(sob5_ave3)
    sob5_ave5 = LineSegmentDetector(sob5_ave5)
    lap3_ave3 = LineSegmentDetector(lap3_ave3)
    lap3_ave5 = LineSegmentDetector(lap3_ave5)
    lap5_ave3 = LineSegmentDetector(lap5_ave3)
    lap5_ave5 = LineSegmentDetector(lap5_ave5)

    can_med3 = CannyEdgeDetector(med3)
    can_med5 = CannyEdgeDetector(med5)
    high_med3 = HighpassFilter(med3)
    high_med5 = HighpassFilter(med5)
    pre3_med3 = PrewittEdgeDetector(med3, ksize=3)
    pre3_med5 = PrewittEdgeDetector(med5, ksize=3)
    pre5_med3 = PrewittEdgeDetector(med3, ksize=5)
    pre5_med5 = PrewittEdgeDetector(med5, ksize=5)
    sob3_med3 = SobelEdgeDetector(med3, ksize=3)
    sob3_med5 = SobelEdgeDetector(med5, ksize=3)
    sob5_med3 = SobelEdgeDetector(med3, ksize=5)
    sob5_med5 = SobelEdgeDetector(med5, ksize=5)
    lap3_med3 = LaplacianEdgeDetector(med3, ksize=3)
    lap3_med5 = LaplacianEdgeDetector(med5, ksize=3)
    lap5_med3 = LaplacianEdgeDetector(med3, ksize=5)
    lap5_med5 = LaplacianEdgeDetector(med5, ksize=5)

    can_gau3 = CannyEdgeDetector(gau3)
    can_gau5 = CannyEdgeDetector(gau5)
    high_gau3 = HighpassFilter(gau3)
    high_gau5 = HighpassFilter(gau5)
    pre3_gau3 = PrewittEdgeDetector(gau3, ksize=3)
    pre3_gau5 = PrewittEdgeDetector(gau5, ksize=3)
    pre5_gau3 = PrewittEdgeDetector(gau3, ksize=5)
    pre5_gau5 = PrewittEdgeDetector(gau5, ksize=5)
    sob3_gau3 = SobelEdgeDetector(gau3, ksize=3)
    sob3_gau5 = SobelEdgeDetector(gau5, ksize=3)
    sob5_gau3 = SobelEdgeDetector(gau3, ksize=5)
    sob5_gau5 = SobelEdgeDetector(gau5, ksize=5)
    lap3_gau3 = LaplacianEdgeDetector(gau3, ksize=3)
    lap3_gau5 = LaplacianEdgeDetector(gau5, ksize=3)
    lap5_gau3 = LaplacianEdgeDetector(gau3, ksize=5)
    lap5_gau5 = LaplacianEdgeDetector(gau5, ksize=5)

    can_bil3 = CannyEdgeDetector(bil3)
    can_bil5 = CannyEdgeDetector(bil5)
    high_bil3 = HighpassFilter(bil3)
    high_bil5 = HighpassFilter(bil5)
    pre3_bil3 = PrewittEdgeDetector(bil3, ksize=3)
    pre3_bil5 = PrewittEdgeDetector(bil5, ksize=3)
    pre5_bil3 = PrewittEdgeDetector(bil3, ksize=5)
    pre5_bil5 = PrewittEdgeDetector(bil5, ksize=5)
    sob3_bil3 = SobelEdgeDetector(bil3, ksize=3)
    sob3_bil5 = SobelEdgeDetector(bil5, ksize=3)
    sob5_bil3 = SobelEdgeDetector(bil3, ksize=5)
    sob5_bil5 = SobelEdgeDetector(bil5, ksize=5)
    lap3_bil3 = LaplacianEdgeDetector(bil3, ksize=3)
    lap3_bil5 = LaplacianEdgeDetector(bil5, ksize=3)
    lap5_bil3 = LaplacianEdgeDetector(bil3, ksize=5)
    lap5_bil5 = LaplacianEdgeDetector(bil5, ksize=5)

    can_low_ifft = CannyEdgeDetector(low_ifft)
    high_low_ifft = HighpassFilter(low_ifft)
    pre3_low_ifft = PrewittEdgeDetector(low_ifft, ksize=3)
    pre5_low_ifft = PrewittEdgeDetector(low_ifft, ksize=5)
    sob3_low_ifft = SobelEdgeDetector(low_ifft, ksize=3)
    sob5_low_ifft = SobelEdgeDetector(low_ifft, ksize=5)
    lap3_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=3)
    lap5_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=5)
    phough_can_ave3 = CannyEdgeDetector(ave3)
    phough_can_ave5 = CannyEdgeDetector(ave5)
    phough_high_ave3 = HighpassFilter(ave3)
    phough_high_ave5 = HighpassFilter(ave5)
    phough_pre3_ave3 = PrewittEdgeDetector(ave3, ksize=3)
    phough_pre3_ave5 = PrewittEdgeDetector(ave5, ksize=3)
    phough_pre5_ave3 = PrewittEdgeDetector(ave3, ksize=5)
    phough_pre5_ave5 = PrewittEdgeDetector(ave5, ksize=5)
    phough_sob3_ave3 = SobelEdgeDetector(ave3, ksize=3)
    phough_sob3_ave5 = SobelEdgeDetector(ave5, ksize=3)
    phough_sob5_ave3 = SobelEdgeDetector(ave3, ksize=5)
    phough_sob5_ave5 = SobelEdgeDetector(ave5, ksize=5)
    phough_lap3_ave3 = LaplacianEdgeDetector(ave3, ksize=3)
    phough_lap3_ave5 = LaplacianEdgeDetector(ave5, ksize=3)
    phough_lap5_ave3 = LaplacianEdgeDetector(ave3, ksize=5)
    phough_lap5_ave5 = LaplacianEdgeDetector(ave5, ksize=5)

    phough_can_med3 = CannyEdgeDetector(med3)
    phough_can_med5 = CannyEdgeDetector(med5)
    phough_high_med3 = HighpassFilter(med3)
    phough_high_med5 = HighpassFilter(med5)
    phough_pre3_med3 = PrewittEdgeDetector(med3, ksize=3)
    phough_pre3_med5 = PrewittEdgeDetector(med5, ksize=3)
    phough_pre5_med3 = PrewittEdgeDetector(med3, ksize=5)
    phough_pre5_med5 = PrewittEdgeDetector(med5, ksize=5)
    phough_sob3_med3 = SobelEdgeDetector(med3, ksize=3)
    phough_sob3_med5 = SobelEdgeDetector(med5, ksize=3)
    phough_sob5_med3 = SobelEdgeDetector(med3, ksize=5)
    phough_sob5_med5 = SobelEdgeDetector(med5, ksize=5)
    phough_lap3_med3 = LaplacianEdgeDetector(med3, ksize=3)
    phough_lap3_med5 = LaplacianEdgeDetector(med5, ksize=3)
    phough_lap5_med3 = LaplacianEdgeDetector(med3, ksize=5)
    phough_lap5_med5 = LaplacianEdgeDetector(med5, ksize=5)

    phough_can_gau3 = CannyEdgeDetector(gau3)
    phough_can_gau5 = CannyEdgeDetector(gau5)
    phough_high_gau3 = HighpassFilter(gau3)
    phough_high_gau5 = HighpassFilter(gau5)
    phough_pre3_gau3 = PrewittEdgeDetector(gau3, ksize=3)
    phough_pre3_gau5 = PrewittEdgeDetector(gau5, ksize=3)
    phough_pre5_gau3 = PrewittEdgeDetector(gau3, ksize=5)
    phough_pre5_gau5 = PrewittEdgeDetector(gau5, ksize=5)
    phough_sob3_gau3 = SobelEdgeDetector(gau3, ksize=3)
    phough_sob3_gau5 = SobelEdgeDetector(gau5, ksize=3)
    phough_sob5_gau3 = SobelEdgeDetector(gau3, ksize=5)
    phough_sob5_gau5 = SobelEdgeDetector(gau5, ksize=5)
    phough_lap3_gau3 = LaplacianEdgeDetector(gau3, ksize=3)
    phough_lap3_gau5 = LaplacianEdgeDetector(gau5, ksize=3)
    phough_lap5_gau3 = LaplacianEdgeDetector(gau3, ksize=5)
    phough_lap5_gau5 = LaplacianEdgeDetector(gau5, ksize=5)

    phough_can_bil3 = CannyEdgeDetector(bil3)
    phough_can_bil5 = CannyEdgeDetector(bil5)
    phough_high_bil3 = HighpassFilter(bil3)
    phough_high_bil5 = HighpassFilter(bil5)
    phough_pre3_bil3 = PrewittEdgeDetector(bil3, ksize=3)
    phough_pre3_bil5 = PrewittEdgeDetector(bil5, ksize=3)
    phough_pre5_bil3 = PrewittEdgeDetector(bil3, ksize=5)
    phough_pre5_bil5 = PrewittEdgeDetector(bil5, ksize=5)
    phough_sob3_bil3 = SobelEdgeDetector(bil3, ksize=3)
    phough_sob3_bil5 = SobelEdgeDetector(bil5, ksize=3)
    phough_sob5_bil3 = SobelEdgeDetector(bil3, ksize=5)
    phough_sob5_bil5 = SobelEdgeDetector(bil5, ksize=5)
    phough_lap3_bil3 = LaplacianEdgeDetector(bil3, ksize=3)
    phough_lap3_bil5 = LaplacianEdgeDetector(bil5, ksize=3)
    phough_lap5_bil3 = LaplacianEdgeDetector(bil3, ksize=5)
    phough_lap5_bil5 = LaplacianEdgeDetector(bil5, ksize=5)

    phough_can_low_ifft = CannyEdgeDetector(low_ifft)
    phough_high_low_ifft = HighpassFilter(low_ifft)
    phough_pre3_low_ifft = PrewittEdgeDetector(low_ifft, ksize=3)
    phough_pre5_low_ifft = PrewittEdgeDetector(low_ifft, ksize=5)
    phough_sob3_low_ifft = SobelEdgeDetector(low_ifft, ksize=3)
    phough_sob5_low_ifft = SobelEdgeDetector(low_ifft, ksize=5)
    phough_lap3_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=3)
    phough_lap5_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=5)

    noise_filters = [AverageFilter(Image, ksize=3), AverageFilter(Image, ksize=5), MedianFilter(Image, ksize=3), MedianFilter(Image, ksize=5), GaussianFilter(Image, ksize=3), GaussianFilter(Image, ksize=5), BilateralFilter(Image, ksize=3), BilateralFilter(Image, ksize=5), IFFT(LowpassFilter(Image, inner_radius=65))]

    edge_filters = [CannyEdgeDetector(filter), IFFT(HighpassFilter(filter)), PrewittEdgeDetector(filter, ksize=3), PrewittEdgeDetector(filter, ksize=5), SobelEdgeDetector(filter, ksize=3), SobelEdgeDetector(filter, ksize=5), LaplacianEdgeDetector(filter, ksize=3), LaplacianEdgeDetector(filter, ksize=5)]

    line_detectiors = [PHoughTransform(filter), HoughTransform(filter), LineSegmentDetector(filter)]

    for i in noise_filters:
        noise_filter = i(original)
        for j in edge_filters:
            edge_filter = j(noise_filter)
            for k in line_detectiors:
                line_detector = k(edge_filter)
                Image.save_image(line_detector._title, line_detector._gray_image)


    # bandpass = BandpassFilter(original, outer_radius=150, inner_radius=50)
    # Image.show_image(bandpass._title, bandpass._fft_image)
    # Image.show_image(bandpass._title, bandpass._masked_fft_image)
    # peak = PeakFilter(bandpass)
    # Image.show_image(peak._title, peak._spot_image)
    # peak_ifft = IFFT(peak)
    # Image.show_image(peak_ifft._title, peak_ifft._gray_image)
    # bin = OtsuBinarization(peak_ifft)
    # compose = ComposeImage(original, bin)
    # Image.show_image(bin._title, bin._gray_image)
    # Image.save_image(compose._title, compose._rgb_image)

