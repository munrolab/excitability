'''
imagestack
'''

import numpy as np

def new_excitation(mask,N):
    '''
    given binary imagestack, calculate difference of images N frames apart

    :param mask: binary imagestack
    :param N: number of frames
    '''
    mask2 = np.zeros(mask.shape,dtype=bool)# new excitations
    temp = mask.astype(float)
    nzzs = np.where(mask.max(1).max(1))[0]
    pzlst = nzzs[N:]
    for i in pzlst:
        mask2[i] = (temp[i]-temp[i-N])>0
    return mask2

def cum(mask,N):
    '''
    given binary imagestack, calculate union of images in the past N frames

    :param mask: binary imagestack
    :param N: number of frames
    '''
    mask2 = np.zeros(mask.shape,dtype=bool)
    nzzs = np.where(mask.max(1).max(1))[0]
    for pz in nzzs[N+1:]:
        mask2[pz] = mask[pz-N:pz].max(0)
    return mask2

def moving_avg(im,N):
    '''
    centered moving average
    out[i] = mean(in[i-N:i+N+1])
    '''
    out = np.zeros(im.shape,dtype=im.dtype)
    for i in range(N,im.shape[0]-N):
        out[i] = im[i-N:i+N+1].mean(0)
    return out

def bleach_correction(im, ixy, window_length, videozmin):
    '''
    measure mean intensity as a function of time, smooth bleach curve, compensate for loss

    :returns: mean intensity over time raw, smoothed, corrected ; corrected imagestack
    :rtype: np.array((N frames,),float), np.array((Z,X,Y),float)
    '''
    from scipy.signal import savgol_filter
    y = im[:,ixy[0],ixy[1]].mean(1)
    ys = savgol_filter(y, window_length=window_length, polyorder=1).astype('uint16')# smooth signal
    compensation = (ys[:videozmin].mean()-ys)[:,None,None]
    imbc = im+compensation
    ybc = imbc[:,ixy[0],ixy[1]].mean(1)
    return y,ys,ybc,imbc

def smooth_binary(y,NMIN):
    '''
    remove short binary segments from timeseries

    This is a simple example::

        from utils import binary_segments

    '''
    from utils import binary_segments
    seglst = binary_segments(y)
    for ii,seg in enumerate(seglst):
        seglst = binary_segments(y)
        if ii==0:
            continue
        if seg[0]:
            if seg[2]<NMIN[0]:
                y[seg[1]:seg[1]+seg[2]] = not seg[0]
        else:
            if seg[2]<NMIN[1]:
                y[seg[1]:seg[1]+seg[2]] = not seg[0]
    return y

