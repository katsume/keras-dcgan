import glob
import os

from PIL import Image

if __name__=='__main__':

    cwd= os.getcwd()

    files= glob.glob(os.path.join(cwd, '*'));
    for file in files:

        dir, name= os.path.split(file)
        base, ext = os.path.splitext(name)
        if not ext in ['.jpg', '.png', '.gif']:
            continue

        img = Image.open(file)
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flipped.save(os.path.join(cwd, base+'_flipped'+ext));

        print(name);
