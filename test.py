from image_processor import Image, Path2Image
from frequency_filter import FFT, IFFT, LowpassFilter
import numpy as np

if __name__ == '__main__':
    # title = "unclear"
    # path = "image/unclear.jpg"
    # title = "clear"
    # path = "clear/clear.jpg"
    title = "test"
    path = "image/NaCl1_noscale.jpg"
    np.set_printoptions(threshold=np.inf)
    original = Path2Image(title, path)
    fft = FFT(original)
    low = LowpassFilter(original, inner_radius=65)
    low_ifft = IFFT(low)
    Image.save_image(fft._title, fft._fft_image, dir='clear/noise')
    Image.save_image(low._title, low._fft_image, dir='clear/noise')
    Image.save_image(low_ifft._title, low_ifft._gray_image, dir='clear/noise')
