import os,sys
from os.path import join as pjoin
sys.path.insert(0, '/Volumes/backup/pyblob')
disk = '/Volumes/backup'

esd = pjoin(disk,sys.argv[1])# txt file containing list of embryos info to analyze
d1 = pjoin(disk,sys.argv[2])# directory of tiffs: phenotype/ID/G.tiff
d2 = pjoin(disk,sys.argv[3])# directory of outlines: ID.tiff
'''
examples:
python scripts/movavg.py tiff_ROK/ani/embryos.txt tiff_ROK tiff_outline
'''

import numpy as np
import read, tifffile
from skimage import io

es = read.readfilelist(esd)

#for ID in es.iloc[:2].index:
for ID in es.index:
    phenotype = es.loc[ID]['phenotype']
    nEnd = int(es.loc[ID]['nEnd'])
    if not os.path.exists(pjoin(d2,'poly-'+ID+'.tif')):
        print(phenotype+'/'+ID)
        d = read.rawtiff(pjoin(d1,phenotype,ID),['G'])
        im = io.imread(d)
        im = im[:nEnd]
        im = np.percentile(im,98,axis=0)
        tifffile.imsave(pjoin(d2,ID+'.tif'), im.astype(np.uint16))
