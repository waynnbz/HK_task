from pathlib import Path

from ImgRead import *
from SHP_SelPoint import *

# set path
curpath = Path.cwd()
# test
curpath = Path.cwd()/'data'

mlipath = curpath / 'MIL'
diffpath = curpath / 'DIFF'
basemapname = 'mli_ave.ras'

# load image data
nlines = 501
mlistack = ImgRead(mlipath, 'mli', nlines, 'float32')
intfstack = ImgRead(diffpath, 'diff', nlines, 'cpxfloat32')

# SHP selection
# CalWin = (15, 15)  # [row col]
# Alpha = 0.05
# [SHP] = SHP_SelPoint(mlistack.datastack, CalWin, Alpha)
