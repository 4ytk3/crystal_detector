from image_processor import Image, OriginalImage
from spatial_filter import AverageFilter, GaussianFilter, MedianFilter, BilateralFilter
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter


if __name__ == '__main__':
    title = "NaCl"
    path = "image/NaCl1_noscale.jpg"
    nacl = OriginalImage(title, path)
    bandpass = BandpassFilter(nacl, outer_radius=120)
    Image.show_image(bandpass._title, bandpass._fft_image)
    peak = PeakFilter(bandpass)
    Image.show_image(peak._title, peak._peak_image)
    Image.show_image(peak._title, peak._spot_image)
    Image.show_image(peak._title, peak._fft_image)
    bandpass_ifft = IFFT(peak)
    Image.show_image(bandpass_ifft._title, bandpass_ifft._ifft_image)