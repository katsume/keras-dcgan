import glob
import os
import argparse

if __name__=='__main__':

    parser= argparse.ArgumentParser()
    parser.add_argument('base', type=str, help='Sync base dir')
    parser.add_argument('target', type=str, help='Sync target dir')
    args= parser.parse_args()

    files= glob.glob(os.path.join(args.target, '*'))
    print('%d files in %s' % (len(files), args.target))
    count= 0
    for file in files:

        dir, basename= os.path.split(file)
        name, ext= os.path.splitext(basename)
        if not glob.glob(os.path.join(args.base, name+'.*')):
            os.unlink(file)
            count+=1

    print('%d files are removed.' % (count));
