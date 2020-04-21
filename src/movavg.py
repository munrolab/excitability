import os,sys
from os.path import join as pjoin
sys.path.insert(0, '/Volumes/backup/pyblob')
disk = '/Volumes/backup'

esd = pjoin(disk,sys.argv[1])# (read)txt file containing list of embryos info to analyze
d1 = pjoin(disk,sys.argv[2])# (read)directory of tiffs: phenotype/ID/G.tif
d2 = pjoin(disk,sys.argv[3])# (read)directory of outlines: poly-ID.tif
d3 = pjoin(disk,sys.argv[4])# (write)directory of moving average: phenotype-ID.tif
'''
examples:
python scripts/movavg.py tiff_ROK/ani/embryos.txt tiff_ROK tiff_outline tiff_ROK_movavg 
'''

import numpy as np
import read, tifffile
from skimage import io

es = read.readfilelist(esd)# read embryos to analyze

from imagestack import moving_avg,bleach_correction

#for ID in es.iloc[::20].index:
for ID in es.index:
    phenotype = es.loc[ID]['phenotype']
    nEnd = int(es.loc[ID]['nEnd'])
    tRes = float(es.loc[ID]['tRes'])
    N = round(1.2/tRes)# !!!!!! temporal smoothing unit is seconds
    NN = 20*N+1# smoothing group size for bleach correction
    
    # correct for photobleach, remove signal from outside of embryo
    outline = io.imread(pjoin(d2,'poly-'+ID+'.tif')).astype(bool)
    x,y = np.where(outline)
    nx,ny = np.where(~outline)

    print(phenotype+'/'+ID)
    d = read.rawtiff(pjoin(d1,phenotype,ID),['G','R'])
    imG = io.imread(d[0])[:nEnd]
    imG = moving_avg(imG, N)
    _,_,imG = bleach_correction(imG,x,y,NN)
    if len(d)==1:
        im = imG
    elif len(d)==2:
        imR = io.imread(d[1])[:nEnd]
        imR = moving_avg(imR, N)
        _,_,imR = bleach_correction(imR,x,y,NN)

        imempty = np.zeros(imG.shape,dtype=imG.dtype)        
        im = np.stack([imG,imR,imempty],axis=3)  
        
    im[:,nx,ny] = 0
    tifffile.imsave(pjoin(d3,phenotype+'-'+ID+'.tif'), im.astype(np.uint16))

