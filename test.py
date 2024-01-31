from image_processor import Image, Path2Image, ComposeImage
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization

if __name__ == '__main__':
    # title = "unclear"
    # path = "image/unclear.jpg"
    # title = "clear"
    # path = "clear/clear.jpg"
    title = "test"
    path = "image/10033_nacl 009.jpg"
    original = Path2Image(title, path)

    bandpass = BandpassFilter(original, outer_radius=100, inner_radius=30)
    Image.show_image(bandpass._title, bandpass._fft_image)
    peak = PeakFilter(bandpass)
    Image.show_image(peak._title, peak._spot_image)
    peak_ifft = IFFT(peak)
    Image.show_image(peak_ifft._title, peak_ifft._gray_image)
    bin = OtsuBinarization(peak_ifft)
    compose = ComposeImage(original, bin)
    Image.show_image(bin._title, bin._gray_image)
    Image.save_image(compose._title, compose._rgb_image)
    Image.pixel_counter(compose._title, compose._rgb_image)