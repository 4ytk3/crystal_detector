from image_processor import Path2Image
from frequency_filter import PeakMask, IFFT
import os
import cv2
import glob
import numpy as np

if __name__ == '__main__':
    images = glob.glob("./NaCl_cluster_video./*.jpg")
    n=0
    for path in images:
        title = "NaCl"
        original = Path2Image(title, path)
        peakmask = PeakMask(original)
        ifft = IFFT(peakmask)
        name = os.path.splitext(os.path.basename(path))[0]
        if n == 0:
            cv2.imwrite(os.path.join('detected', f'fft.jpg'), peakmask._masked_fft_image.astype(np.float32))
        cv2.imwrite(os.path.join('detected', f'{name}.jpg'), ifft._gray_image.astype(np.float32))
        n+=1
