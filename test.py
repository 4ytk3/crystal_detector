from image_processor import Image, Path2Image, ComposeImage
from frequency_filter import FFT, IFFT, LowpassFilter, HighpassFilter, BandpassFilter, PeakFilter
from binarization import OtsuBinarization

if __name__ == '__main__':
    title = "unclear"
    path = "unclear/unclear.jpg"
    original = Path2Image(title, path)

    bandpass = BandpassFilter(original, outer_radius=60, inner_radius=55)
    ifft = IFFT(bandpass)
    Image.show_image(bandpass._title, bandpass._fft_image)
    Image.show_image(ifft._title, ifft._gray_image)

    lowpass = LowpassFilter(original, inner_radius=55)
    low_ifft = IFFT(lowpass)
    Image.show_image(lowpass._title, lowpass._fft_image)
    Image.show_image(low_ifft._title, low_ifft._gray_image)

    highpass = HighpassFilter(lowpass, outer_radius=60)
    high_ifft = IFFT(highpass)
    Image.show_image(highpass._title, highpass._fft_image)
    Image.show_image(high_ifft._title, high_ifft._gray_image)

    # peak = PeakFilter(bandpass)
    # Image.show_image(peak._title, peak._spot_image)
    # peak_ifft = IFFT(peak)
    # Image.show_image(peak_ifft._title, peak_ifft._gray_image)
    # bin = OtsuBinarization(peak_ifft)
    # compose = ComposeImage(original, bin)
    # Image.show_image(bin._title, bin._gray_image)
    # Image.save_image(compose._title, compose._rgb_image)
    # Image.pixel_counter(compose._title, compose._rgb_image)