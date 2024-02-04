import os
import glob

if __name__ == "__main__":
    folder = 'nacl'
    files = glob.glob(folder + '/*')
    n=1
    for f in files:
        os.rename(f, os.path.join(folder, f"{str(n).zfill(3)}.jpg"))
        n+=1