from image_processor import Image, Path2Image, ComposeImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization
from edge_detector import LaplacianEdgeDetector, CannyEdgeDetector, PrewittEdgeDetector, SobelEdgeDetector
from line_detector import HoughTransform, PHoughTransform, LineSegmentDetector
import cv2
import statistics

if __name__ == '__main__':
    title = "unclear"
    path = "unclear/unclear.jpg"
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
    fft = FFT(original)
    low = LowpassFilter(original, inner_radius=65)
    low_ifft = IFFT(low)
    # Image.save_image(ave3._title, ave3._gray_image, dir='unclear/noise')
    # Image.save_image(ave5._title, ave5._gray_image, dir='unclear/noise')
    # Image.save_image(med3._title, med3._gray_image, dir='unclear/noise')
    # Image.save_image(med5._title, med5._gray_image, dir='unclear/noise')
    # Image.save_image(gau3._title, gau3._gray_image, dir='unclear/noise')
    # Image.save_image(gau5._title, gau5._gray_image, dir='unclear/noise')
    # Image.save_image(bil3._title, bil3._gray_image, dir='unclear/noise')
    # Image.save_image(bil5._title, bil5._gray_image, dir='unclear/noise')
    # Image.save_image(fft._title, fft._fft_image, dir='unclear/noise')
    # Image.save_image(low._title, low._fft_image, dir='unclear/noise')
    # Image.save_image(low_ifft._title, low_ifft._gray_image, dir='unclear/noise')


    # canny
    can_ave3 = CannyEdgeDetector(ave3)
    can_ave5 = CannyEdgeDetector(ave5)
    can_med3 = CannyEdgeDetector(med3)
    can_med5 = CannyEdgeDetector(med5)
    can_gau3 = CannyEdgeDetector(gau3)
    can_gau5 = CannyEdgeDetector(gau5)
    can_bil3 = CannyEdgeDetector(bil3)
    can_bil5 = CannyEdgeDetector(bil5)
    can_low_ifft = CannyEdgeDetector(low_ifft)

    cans = [can_ave3,can_ave5,can_med3,can_med5,can_gau3,can_gau5,can_bil3,can_bil5,can_low_ifft]

    for i in cans:
        Image.save_image(i._title, i._gray_image, dir='unclear/canny')

    # high
    high_ave3 = HighpassFilter(ave3, outer_radius=55)
    high_ave3_ifft = IFFT(high_ave3)
    high_ave5 = HighpassFilter(ave5, outer_radius=55)
    high_ave5_ifft = IFFT(high_ave5)
    high_med3 = HighpassFilter(med3, outer_radius=55)
    high_med3_ifft = IFFT(high_med3)
    high_med5 = HighpassFilter(med5, outer_radius=55)
    high_med5_ifft = IFFT(high_med5)
    high_gau3 = HighpassFilter(gau3, outer_radius=55)
    high_gau3_ifft = IFFT(high_gau3)
    high_gau5 = HighpassFilter(gau5, outer_radius=55)
    high_gau5_ifft = IFFT(high_gau5)
    high_bil3 = HighpassFilter(bil3, outer_radius=55)
    high_bil3_ifft = IFFT(high_bil3)
    high_bil5 = HighpassFilter(bil5, outer_radius=55)
    high_bil5_ifft = IFFT(high_bil5)
    high_low_ifft = HighpassFilter(low_ifft, outer_radius=55)
    high_low_ifft_ifft = IFFT(high_low_ifft)

    highs = [high_ave3_ifft,high_ave5_ifft,high_med3_ifft,high_med5_ifft,high_gau3_ifft,high_gau5_ifft,high_bil3_ifft,high_bil5_ifft,high_low_ifft_ifft]

    for i in highs:
        Image.save_image(i._title, i._gray_image, dir='unclear/highpass')

    high_masks = [high_ave3,high_ave5,high_med3,high_med5,high_gau3,high_gau5,high_bil3,high_bil5,high_low_ifft]

    for i in high_masks:
        Image.save_image(i._title, i._fft_image, dir='unclear/highpass')

    # prewitt
    pre3_ave3 = PrewittEdgeDetector(ave3, ksize=3)
    pre3_ave5 = PrewittEdgeDetector(ave5, ksize=3)
    pre5_ave3 = PrewittEdgeDetector(ave3, ksize=5)
    pre5_ave5 = PrewittEdgeDetector(ave5, ksize=5)
    pre3_med3 = PrewittEdgeDetector(med3, ksize=3)
    pre3_med5 = PrewittEdgeDetector(med5, ksize=3)
    pre5_med3 = PrewittEdgeDetector(med3, ksize=5)
    pre5_med5 = PrewittEdgeDetector(med5, ksize=5)
    pre3_gau3 = PrewittEdgeDetector(gau3, ksize=3)
    pre3_gau5 = PrewittEdgeDetector(gau5, ksize=3)
    pre5_gau3 = PrewittEdgeDetector(gau3, ksize=5)
    pre5_gau5 = PrewittEdgeDetector(gau5, ksize=5)
    pre3_bil3 = PrewittEdgeDetector(bil3, ksize=3)
    pre3_bil5 = PrewittEdgeDetector(bil5, ksize=3)
    pre5_bil3 = PrewittEdgeDetector(bil3, ksize=5)
    pre5_bil5 = PrewittEdgeDetector(bil5, ksize=5)
    pre3_low_ifft = PrewittEdgeDetector(low_ifft, ksize=3)
    pre5_low_ifft = PrewittEdgeDetector(low_ifft, ksize=5)

    pres = [pre3_ave3,pre3_ave5,pre5_ave3,pre5_ave5,pre3_med3,pre3_med5,pre5_med3,pre5_med5,pre3_gau3,pre3_gau5,pre5_gau3,pre5_gau5,pre3_bil3,pre3_bil5,pre5_bil3,pre5_bil5,pre3_low_ifft,pre5_low_ifft]
    for i in pres:
        Image.save_image(i._title, i._gray_image, dir='unclear/prewitt')

    # sobel
    sob3_ave3 = SobelEdgeDetector(ave3, ksize=3)
    sob3_ave5 = SobelEdgeDetector(ave5, ksize=3)
    sob5_ave3 = SobelEdgeDetector(ave3, ksize=5)
    sob5_ave5 = SobelEdgeDetector(ave5, ksize=5)
    sob3_med3 = SobelEdgeDetector(med3, ksize=3)
    sob3_med5 = SobelEdgeDetector(med5, ksize=3)
    sob5_med3 = SobelEdgeDetector(med3, ksize=5)
    sob5_med5 = SobelEdgeDetector(med5, ksize=5)
    sob3_gau3 = SobelEdgeDetector(gau3, ksize=3)
    sob3_gau5 = SobelEdgeDetector(gau5, ksize=3)
    sob5_gau3 = SobelEdgeDetector(gau3, ksize=5)
    sob5_gau5 = SobelEdgeDetector(gau5, ksize=5)
    sob3_bil3 = SobelEdgeDetector(bil3, ksize=3)
    sob3_bil5 = SobelEdgeDetector(bil5, ksize=3)
    sob5_bil3 = SobelEdgeDetector(bil3, ksize=5)
    sob5_bil5 = SobelEdgeDetector(bil5, ksize=5)
    sob3_low_ifft = SobelEdgeDetector(low_ifft, ksize=3)
    sob5_low_ifft = SobelEdgeDetector(low_ifft, ksize=5)

    sobs = [sob3_ave3,sob3_ave5,sob5_ave3,sob5_ave5,sob3_med3,sob3_med5,sob5_med3,sob5_med5,sob3_gau3,sob3_gau5,sob5_gau3,sob5_gau5,sob3_bil3,sob3_bil5,sob5_bil3,sob5_bil5,sob3_low_ifft,sob5_low_ifft]
    for i in sobs:
        Image.save_image(i._title, i._gray_image, dir='unclear/sobel')

    # lap
    lap3_ave3 = LaplacianEdgeDetector(ave3, ksize=3)
    lap3_ave5 = LaplacianEdgeDetector(ave5, ksize=3)
    lap5_ave3 = LaplacianEdgeDetector(ave3, ksize=5)
    lap5_ave5 = LaplacianEdgeDetector(ave5, ksize=5)
    lap3_med3 = LaplacianEdgeDetector(med3, ksize=3)
    lap3_med5 = LaplacianEdgeDetector(med5, ksize=3)
    lap5_med3 = LaplacianEdgeDetector(med3, ksize=5)
    lap5_med5 = LaplacianEdgeDetector(med5, ksize=5)
    lap3_gau3 = LaplacianEdgeDetector(gau3, ksize=3)
    lap3_gau5 = LaplacianEdgeDetector(gau5, ksize=3)
    lap5_gau3 = LaplacianEdgeDetector(gau3, ksize=5)
    lap5_gau5 = LaplacianEdgeDetector(gau5, ksize=5)
    lap3_bil3 = LaplacianEdgeDetector(bil3, ksize=3)
    lap3_bil5 = LaplacianEdgeDetector(bil5, ksize=3)
    lap5_bil3 = LaplacianEdgeDetector(bil3, ksize=5)
    lap5_bil5 = LaplacianEdgeDetector(bil5, ksize=5)
    lap3_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=3)
    lap5_low_ifft = LaplacianEdgeDetector(low_ifft, ksize=5)

    laps = [lap3_ave3,lap3_ave5,lap5_ave3,lap5_ave5,lap3_med3,lap3_med5,lap5_med3,lap5_med5,lap3_gau3,lap3_gau5,lap5_gau3,lap5_gau5,lap3_bil3,lap3_bil5,lap5_bil3,lap5_bil5,lap3_low_ifft,lap5_low_ifft]
    for i in laps:
        Image.save_image(i._title, i._gray_image, dir='unclear/laplacian')

    # phough
    phough_can_ave3 = PHoughTransform(can_ave3)
    phough_can_ave5 = PHoughTransform(can_ave5)
    phough_high_ave3 = PHoughTransform(high_ave3_ifft)
    phough_high_ave5 = PHoughTransform(high_ave5_ifft)
    phough_pre3_ave3 = PHoughTransform(pre3_ave3)
    phough_pre3_ave5 = PHoughTransform(pre3_ave5)
    phough_pre5_ave3 = PHoughTransform(pre5_ave3)
    phough_pre5_ave5 = PHoughTransform(pre5_ave5)
    phough_sob3_ave3 = PHoughTransform(sob3_ave3)
    phough_sob3_ave5 = PHoughTransform(sob3_ave5)
    phough_sob5_ave3 = PHoughTransform(sob5_ave3)
    phough_sob5_ave5 = PHoughTransform(sob5_ave5)
    phough_lap3_ave3 = PHoughTransform(lap3_ave3)
    phough_lap3_ave5 = PHoughTransform(lap3_ave5)
    phough_lap5_ave3 = PHoughTransform(lap5_ave3)
    phough_lap5_ave5 = PHoughTransform(lap5_ave5)

    phough_can_med3 = PHoughTransform(can_med3)
    phough_can_med5 = PHoughTransform(can_med5)
    phough_high_med3 = PHoughTransform(high_med3_ifft)
    phough_high_med5 = PHoughTransform(high_med5_ifft)
    phough_pre3_med3 = PHoughTransform(pre3_med3)
    phough_pre3_med5 = PHoughTransform(pre3_med5)
    phough_pre5_med3 = PHoughTransform(pre5_med3)
    phough_pre5_med5 = PHoughTransform(pre5_med5)
    phough_sob3_med3 = PHoughTransform(sob3_med3)
    phough_sob3_med5 = PHoughTransform(sob3_med5)
    phough_sob5_med3 = PHoughTransform(sob5_med3)
    phough_sob5_med5 = PHoughTransform(sob5_med5)
    phough_lap3_med3 = PHoughTransform(lap3_med3)
    phough_lap3_med5 = PHoughTransform(lap3_med5)
    phough_lap5_med3 = PHoughTransform(lap5_med3)
    phough_lap5_med5 = PHoughTransform(lap5_med5)

    phough_can_gau3 = PHoughTransform(can_gau3)
    phough_can_gau5 = PHoughTransform(can_gau5)
    phough_high_gau3 = PHoughTransform(high_gau3_ifft)
    phough_high_gau5 = PHoughTransform(high_gau5_ifft)
    phough_pre3_gau3 = PHoughTransform(pre3_gau3)
    phough_pre3_gau5 = PHoughTransform(pre3_gau5)
    phough_pre5_gau3 = PHoughTransform(pre5_gau3)
    phough_pre5_gau5 = PHoughTransform(pre5_gau5)
    phough_sob3_gau3 = PHoughTransform(sob3_gau3)
    phough_sob3_gau5 = PHoughTransform(sob3_gau5)
    phough_sob5_gau3 = PHoughTransform(sob5_gau3)
    phough_sob5_gau5 = PHoughTransform(sob5_gau5)
    phough_lap3_gau3 = PHoughTransform(lap3_gau3)
    phough_lap3_gau5 = PHoughTransform(lap3_gau5)
    phough_lap5_gau3 = PHoughTransform(lap5_gau3)
    phough_lap5_gau5 = PHoughTransform(lap5_gau5)

    phough_can_bil3 = PHoughTransform(can_bil3)
    phough_can_bil5 = PHoughTransform(can_bil5)
    phough_high_bil3 = PHoughTransform(high_bil3_ifft)
    phough_high_bil5 = PHoughTransform(high_bil5_ifft)
    phough_pre3_bil3 = PHoughTransform(pre3_bil3)
    phough_pre3_bil5 = PHoughTransform(pre3_bil5)
    phough_pre5_bil3 = PHoughTransform(pre5_bil3)
    phough_pre5_bil5 = PHoughTransform(pre5_bil5)
    phough_sob3_bil3 = PHoughTransform(sob3_bil3)
    phough_sob3_bil5 = PHoughTransform(sob3_bil5)
    phough_sob5_bil3 = PHoughTransform(sob5_bil3)
    phough_sob5_bil5 = PHoughTransform(sob5_bil5)
    phough_lap3_bil3 = PHoughTransform(lap3_bil3)
    phough_lap3_bil5 = PHoughTransform(lap3_bil5)
    phough_lap5_bil3 = PHoughTransform(lap5_bil3)
    phough_lap5_bil5 = PHoughTransform(lap5_bil5)

    phough_can_low_ifft = PHoughTransform(can_low_ifft)
    phough_high_low_ifft = PHoughTransform(high_low_ifft_ifft)
    phough_pre3_low_ifft = PHoughTransform(pre3_low_ifft)
    phough_pre5_low_ifft = PHoughTransform(pre5_low_ifft)
    phough_sob3_low_ifft = PHoughTransform(sob3_low_ifft)
    phough_sob5_low_ifft = PHoughTransform(sob5_low_ifft)
    phough_lap3_low_ifft = PHoughTransform(lap3_low_ifft)
    phough_lap5_low_ifft = PHoughTransform(lap5_low_ifft)

    # LSD
    lsd_can_ave3 = LineSegmentDetector(can_ave3)
    lsd_can_ave5 = LineSegmentDetector(can_ave5)
    lsd_high_ave3 = LineSegmentDetector(high_ave3_ifft)
    lsd_high_ave5 = LineSegmentDetector(high_ave5_ifft)
    lsd_pre3_ave3 = LineSegmentDetector(pre3_ave3)
    lsd_pre3_ave5 = LineSegmentDetector(pre3_ave5)
    lsd_pre5_ave3 = LineSegmentDetector(pre5_ave3)
    lsd_pre5_ave5 = LineSegmentDetector(pre5_ave5)
    lsd_sob3_ave3 = LineSegmentDetector(sob3_ave3)
    lsd_sob3_ave5 = LineSegmentDetector(sob3_ave5)
    lsd_sob5_ave3 = LineSegmentDetector(sob5_ave3)
    lsd_sob5_ave5 = LineSegmentDetector(sob5_ave5)
    lsd_lap3_ave3 = LineSegmentDetector(lap3_ave3)
    lsd_lap3_ave5 = LineSegmentDetector(lap3_ave5)
    lsd_lap5_ave3 = LineSegmentDetector(lap5_ave3)
    lsd_lap5_ave5 = LineSegmentDetector(lap5_ave5)

    lsd_can_med3 = LineSegmentDetector(can_med3)
    lsd_can_med5 = LineSegmentDetector(can_med5)
    lsd_high_med3 = LineSegmentDetector(high_med3_ifft)
    lsd_high_med5 = LineSegmentDetector(high_med5_ifft)
    lsd_pre3_med3 = LineSegmentDetector(pre3_med3)
    lsd_pre3_med5 = LineSegmentDetector(pre3_med5)
    lsd_pre5_med3 = LineSegmentDetector(pre5_med3)
    lsd_pre5_med5 = LineSegmentDetector(pre5_med5)
    lsd_sob3_med3 = LineSegmentDetector(sob3_med3)
    lsd_sob3_med5 = LineSegmentDetector(sob3_med5)
    lsd_sob5_med3 = LineSegmentDetector(sob5_med3)
    lsd_sob5_med5 = LineSegmentDetector(sob5_med5)
    lsd_lap3_med3 = LineSegmentDetector(lap3_med3)
    lsd_lap3_med5 = LineSegmentDetector(lap3_med5)
    lsd_lap5_med3 = LineSegmentDetector(lap5_med3)
    lsd_lap5_med5 = LineSegmentDetector(lap5_med5)

    lsd_can_gau3 = LineSegmentDetector(can_gau3)
    lsd_can_gau5 = LineSegmentDetector(can_gau5)
    lsd_high_gau3 = LineSegmentDetector(high_gau3_ifft)
    lsd_high_gau5 = LineSegmentDetector(high_gau5_ifft)
    lsd_pre3_gau3 = LineSegmentDetector(pre3_gau3)
    lsd_pre3_gau5 = LineSegmentDetector(pre3_gau5)
    lsd_pre5_gau3 = LineSegmentDetector(pre5_gau3)
    lsd_pre5_gau5 = LineSegmentDetector(pre5_gau5)
    lsd_sob3_gau3 = LineSegmentDetector(sob3_gau3)
    lsd_sob3_gau5 = LineSegmentDetector(sob3_gau5)
    lsd_sob5_gau3 = LineSegmentDetector(sob5_gau3)
    lsd_sob5_gau5 = LineSegmentDetector(sob5_gau5)
    lsd_lap3_gau3 = LineSegmentDetector(lap3_gau3)
    lsd_lap3_gau5 = LineSegmentDetector(lap3_gau5)
    lsd_lap5_gau3 = LineSegmentDetector(lap5_gau3)
    lsd_lap5_gau5 = LineSegmentDetector(lap5_gau5)

    lsd_can_bil3 = LineSegmentDetector(can_bil3)
    lsd_can_bil5 = LineSegmentDetector(can_bil5)
    lsd_high_bil3 = LineSegmentDetector(high_bil3_ifft)
    lsd_high_bil5 = LineSegmentDetector(high_bil5_ifft)
    lsd_pre3_bil3 = LineSegmentDetector(pre3_bil3)
    lsd_pre3_bil5 = LineSegmentDetector(pre3_bil5)
    lsd_pre5_bil3 = LineSegmentDetector(pre5_bil3)
    lsd_pre5_bil5 = LineSegmentDetector(pre5_bil5)
    lsd_sob3_bil3 = LineSegmentDetector(sob3_bil3)
    lsd_sob3_bil5 = LineSegmentDetector(sob3_bil5)
    lsd_sob5_bil3 = LineSegmentDetector(sob5_bil3)
    lsd_sob5_bil5 = LineSegmentDetector(sob5_bil5)
    lsd_lap3_bil3 = LineSegmentDetector(lap3_bil3)
    lsd_lap3_bil5 = LineSegmentDetector(lap3_bil5)
    lsd_lap5_bil3 = LineSegmentDetector(lap5_bil3)
    lsd_lap5_bil5 = LineSegmentDetector(lap5_bil5)

    lsd_can_low_ifft = LineSegmentDetector(can_low_ifft)
    lsd_high_low_ifft = LineSegmentDetector(high_low_ifft_ifft)
    lsd_pre3_low_ifft = LineSegmentDetector(pre3_low_ifft)
    lsd_pre5_low_ifft = LineSegmentDetector(pre5_low_ifft)
    lsd_sob3_low_ifft = LineSegmentDetector(sob3_low_ifft)
    lsd_sob5_low_ifft = LineSegmentDetector(sob5_low_ifft)
    lsd_lap3_low_ifft = LineSegmentDetector(lap3_low_ifft)
    lsd_lap5_low_ifft = LineSegmentDetector(lap5_low_ifft)

    phoughs = [phough_can_ave3,phough_can_ave5,phough_high_ave3,phough_high_ave5,phough_pre3_ave3,phough_pre3_ave5,phough_pre5_ave3,phough_pre5_ave5,phough_sob3_ave3,phough_sob3_ave5,phough_sob5_ave3,phough_sob5_ave5,phough_lap3_ave3,phough_lap3_ave5,phough_lap5_ave3,phough_lap5_ave5,phough_can_med3,phough_can_med5,phough_high_med3,phough_high_med5,phough_pre3_med3,phough_pre3_med5,phough_pre5_med3,phough_pre5_med5,phough_sob3_med3,phough_sob3_med5,phough_sob5_med3,phough_sob5_med5,phough_lap3_med3,phough_lap3_med5,phough_lap5_med3,phough_lap5_med5,phough_can_gau3,phough_can_gau5,phough_high_gau3,phough_high_gau5,phough_pre3_gau3,phough_pre3_gau5,phough_pre5_gau3,phough_pre5_gau5,phough_sob3_gau3,phough_sob3_gau5,phough_sob5_gau3,phough_sob5_gau5,phough_lap3_gau3,phough_lap3_gau5,phough_lap5_gau3,phough_lap5_gau5,phough_can_bil3,phough_can_bil5,phough_high_bil3,phough_high_bil5,phough_pre3_bil3,phough_pre3_bil5,phough_pre5_bil3,phough_pre5_bil5,phough_sob3_bil3,phough_sob3_bil5,phough_sob5_bil3,phough_sob5_bil5,phough_lap3_bil3,phough_lap3_bil5,phough_lap5_bil3,phough_lap5_bil5,phough_can_low_ifft,phough_high_low_ifft,phough_pre3_low_ifft,phough_pre5_low_ifft,phough_sob3_low_ifft,phough_sob5_low_ifft,phough_lap3_low_ifft,phough_lap5_low_ifft]

    lsds = [lsd_can_ave3,lsd_can_ave5,lsd_high_ave3,lsd_high_ave5,lsd_pre3_ave3,lsd_pre3_ave5,lsd_pre5_ave3,lsd_pre5_ave5,lsd_sob3_ave3,lsd_sob3_ave5,lsd_sob5_ave3,lsd_sob5_ave5,lsd_lap3_ave3,lsd_lap3_ave5,lsd_lap5_ave3,lsd_lap5_ave5,lsd_can_med3,lsd_can_med5,lsd_high_med3,lsd_high_med5,lsd_pre3_med3,lsd_pre3_med5,lsd_pre5_med3,lsd_pre5_med5,lsd_sob3_med3,lsd_sob3_med5,lsd_sob5_med3,lsd_sob5_med5,lsd_lap3_med3,lsd_lap3_med5,lsd_lap5_med3,lsd_lap5_med5,lsd_can_gau3,lsd_can_gau5,lsd_high_gau3,lsd_high_gau5,lsd_pre3_gau3,lsd_pre3_gau5,lsd_pre5_gau3,lsd_pre5_gau5,lsd_sob3_gau3,lsd_sob3_gau5,lsd_sob5_gau3,lsd_sob5_gau5,lsd_lap3_gau3,lsd_lap3_gau5,lsd_lap5_gau3,lsd_lap5_gau5,lsd_can_bil3,lsd_can_bil5,lsd_high_bil3,lsd_high_bil5,lsd_pre3_bil3,lsd_pre3_bil5,lsd_pre5_bil3,lsd_pre5_bil5,lsd_sob3_bil3,lsd_sob3_bil5,lsd_sob5_bil3,lsd_sob5_bil5,lsd_lap3_bil3,lsd_lap3_bil5,lsd_lap5_bil3,lsd_lap5_bil5,lsd_can_low_ifft,lsd_high_low_ifft,lsd_pre3_low_ifft,lsd_pre5_low_ifft,lsd_sob3_low_ifft,lsd_sob5_low_ifft,lsd_lap3_low_ifft,lsd_lap5_low_ifft]

    for i in phoughs:
        try:
            compose = ComposeImage(original, i)
            Image.save_image(i._title, compose._rgb_image, dir='unclear/phough')
        except AttributeError as e:
            print(e)
            print(f"{i._title}")
            pass

    for i in lsds:
        try:
            compose = ComposeImage(original, i)
            Image.save_image(i._title, compose._rgb_image, dir='unclear/LSD')
        except AttributeError as e:
            print(e)
            print(f"{i._title}")
            pass

    # bandpass = BandpassFilter(original, outer_radius=150, inner_radius=50)
    # Image.show_image(bandpass._title, bandpass._fft_image)
    # Image.show_image(bandpass._title, bandpass._fft_image)
    # peak = PeakFilter(bandpass)
    # Image.show_image(peak._title, peak._spot_image)
    # peak_ifft = IFFT(peak)
    # Image.show_image(peak_ifft._title, peak_ifft._gray_image)
    # bin = OtsuBinarization(peak_ifft)
    # compose = ComposeImage(original, bin)
    # Image.show_image(bin._title, bin._gray_image)
    # Image.save_image(compose._title, compose._rgb_image)

