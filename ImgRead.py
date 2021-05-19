from pathlib import Path
import time


def ImgRead(imgpath, suffixname, nline,
            bkformat='float32', machinefmt='b'):
    '''

    :param imgpath:
    :param suffixname:
    :param nline:
    :param bkformat:
    :param machinefmt:
    :return:
    '''

    imgpath = Path(imgpath)

    tag_files =list(imgpath.glob('*'+suffixname))
    img_num = len(tag_files)
    print(f'The number of the {suffixname} images: {img_num}')

    for ii in range(img_num):
        t = time.time()

        


        elapsed = time.time() - t
        print(f'Reading Img {ii}/{img_num}, time = {elapsed} sec')
    return Data