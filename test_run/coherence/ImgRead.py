import numpy as np
import re
from pathlib import Path
import time

from bash_function import read_file


def ImgRead(files, nlines, dtype='float'):

    stack = []
    s_list = []

    for i, file in enumerate(files):
        t = time.time()
        data, _ = read_file(file, nlines, types=dtype)
        stack.append(data)
        temp = re.findall(r'\d+', file)
        if len(temp) == 1:
            s_list.append(float(temp[0]))
        elif len(temp) == 2:
            s_list.append([float(temp[0]), float(temp[1])])
        else:
            print("The format of file name should be: <yyyymmdd> or <yyyymmdd_yyyymmdd>")

        elapsed = time.time() - t
        print(f'Reading Img {i+1} / {len(files)}, time = {elapsed} sec\n')

    stack = np.stack(stack, axis=2)
    s_list = np.array(s_list, ndmin=2)

    return stack, s_list