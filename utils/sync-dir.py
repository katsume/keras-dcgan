import glob
import os
import argparse

if __name__=='__main__':

    parser= argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='Sync source dir')
    parser.add_argument('target', type=str, help='Sync target dir')
    args= parser.parse_args()

    files= glob.glob(os.path.join(args.target, '*'));
    print('%d files in %s' % (len(files), args.target));
    count= 0;
    for file in files:

        dir, name= os.path.split(file)
        if not os.path.exists(os.path.join(args.src, name)):
            os.unlink(file);
            count++;

    print('%d files are removed.' % (count));
