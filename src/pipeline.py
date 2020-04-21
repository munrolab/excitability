'''
pipeline of analysis
'''
from skimage import io
import numpy as np
from os.path import join as pjoin

def movavg_bc(path,channel='G',dt=1.2,gauss=None):
    '''
    calculate moving average in z; correct for photobleach
    '''
    global tRes, outlinexy, ID, phenotype, videotmin, nEnd, nzzs
    from imagestack import moving_avg,bleach_correction
    from utils import rawtiff
    import cv2
    im = io.imread(rawtiff(path,[channel]))[round(2.4/tRes):nEnd]
    im = moving_avg(im, round(dt/tRes))
    nzzs = np.where(im.max(1).max(1))[0]# non zero z positions
    y,ys,ybc,im[nzzs] = bleach_correction(im[nzzs],outlinexy, round(40/tRes)*2+1, round(videotmin/tRes))
    if gauss is not None:
        for i in nzzs:
            im[i] = cv2.blur(im[i],(gauss,gauss))
    return im,y,ys,ybc

def diff(im,tdiff,tsmooth):
    '''
    normalize im between 0 and 1; smoothe in xy using guidedFilter; temporal difference; smoothe in z using savgol_filter

    :param tdiff,tsmooth: time window in seconds for difference and smooth in time
    '''
    global tRes, outlinexy, nzzs
    from cv2.ximgproc import guidedFilter
    from scipy.signal import savgol_filter
    
    # normalize image
    IMG = im.astype(np.float32)
    arr = IMG[:,outlinexy[0],outlinexy[1]]
    arr = arr[arr>0]
    UB = np.percentile(arr,99.999)
    IMG[IMG>UB] = UB
    IMG /= UB

    IMGSM = np.zeros(IMG.shape,dtype=float)# smooth image
    for i in nzzs:
        arr = IMG[i,:,:]
        IMGSM[i,:,:] = guidedFilter(arr,arr,5,0.1)
    IMGSM[nzzs] = savgol_filter(IMGSM[nzzs], round(tsmooth/tRes)*2+1, 2, axis=0)

    IMGD = np.zeros(IMG.shape,dtype=float)
    ndiff = round(tdiff/tRes)
    for i in nzzs[ndiff:]:
        IMGD[i] = IMGSM[i]-IMGSM[i-ndiff]

    return IMG.astype(np.float32),IMGSM.astype(np.float32),IMGD.astype(np.float32)

def excitation(mask,gaptmin=[6,3]):
    '''
    remove short(in time) and small(in area) excitations
    find pixels that enter from non-excited to excited state

    :param mask: binary mask where diff is larger than threshold
    :type mask: np.array((Z,X,Y),bool)
    :return: binary array of excitations and new_excitations. both of type np.array((Z,X,Y),bool)
    '''
    global tRes, nzzs
    from skimage.morphology import remove_small_objects
    from imagestack import smooth_binary
    gapN = np.array(gaptmin)/tRes
    imen = mask.sum(0)
    idxx,idxy = np.where(imen>1)
    for i in range(len(idxx)):
        ts = mask[nzzs, idxx[i], idxy[i]]
        mask[nzzs, idxx[i], idxy[i]] = smooth_binary(ts, gapN)
    for i in nzzs:
        remove_small_objects(mask[i], min_size=25, in_place=True)
    return mask

def save_rois(path,graph,cntdict):
    from lib.hdf5 import save_dict_to_hdf5
    h5dict = {}
    for (pz,i),(parents,children,LABEL) in graph.items():
        roi_id = str(len(h5dict))
        cnt = cntdict[pz][i]
        name = str(LABEL)
        h5dict[roi_id] = {'name':name,'z':pz,
                          'cntx':cnt[:,0,0],'cnty':cnt[:,0,1]}
    save_dict_to_hdf5(h5dict, path, mode='w')

def periodicity(newexc,FMAX,TMAX,DT=2.4):
    '''
    use imagestack of new excitation to calculate temporal frequency and wait time distribution between consecutative excitations
    
    :param newexc: imagestack
    :type newexc: np.array((Z,X,Y),bool)
    :param FMAX: max value for frequency
    :param TMAX: max value for wait time
    :return: newexcsum,(frequency_t,frequency_y),(wait_t,wait_y)
    '''
    global tRes, outline, outlinexy
    newexcsum = newexc.sum(0).astype(float)
    newexcsum[~outline]=np.nan
    arr = newexcsum[outline]
    arr[arr>FMAX] = FMAX
    result,edges = np.histogram(arr,bins=np.arange(-0.5,FMAX+0.5),density=True)
    frequency = (np.arange(0,FMAX), result*(edges[1]-edges[0]))

    newexc = newexc[:,outlinexy[0],outlinexy[1]]
    N = newexc.shape[1]
    mask = newexc.sum(0)>=2
    arr = []
    for idx in np.where(mask)[0]:
        y = newexc[:,idx]
        w = np.nonzero(y)[0]
        w = np.diff(w)*tRes
        arr += list(w)
    arr = np.array(arr)
    arr[arr>TMAX] = TMAX
    result,edges = np.histogram(arr, bins=np.arange(0,TMAX+DT,DT), density=True)
    wait = (edges[1:], result*(edges[1]-edges[0]))
    return newexcsum,frequency,wait

def kinetics(points,imstack,T,radius=7):
    '''
    measure mean intensity in circles as a functin of time
    
    :return: np.array((Ncircles, time, channel))
    '''
    if len(imstack.shape)==3:
        imstack = np.expand_dims(imstack,3)
    global tRes, nzzs
    from skimage.draw import circle
    PRE,POST = round(T[0]/tRes),round((T[-1]+1)/tRes)
    zrange = np.arange(PRE,POST).astype(int)
    
    MIN,MAX = nzzs[0],nzzs[-1]
    mask = np.logical_and(MIN<=points[:,0]+PRE,points[:,0]+POST<=MAX)
    points = points[mask].astype(int)
    if len(points)==0:
        return points,None,None
    
    y = np.zeros((len(points),len(zrange),imstack.shape[-1]))
    rr,cc = circle(0,0,radius)
    for i,(cz,cx,cy) in enumerate(points):
        y[i] = imstack[cz+zrange][:,rr+cx,cc+cy].mean(1)#average of all pixels in circle
    
    from scipy.interpolate import interp2d
    t = np.arange(T[0],T[-1])
    Y = np.zeros((y.shape[0],len(t),imstack.shape[-1]))
    prange = np.arange(len(points))
    for i in range(imstack.shape[-1]):
        f = interp2d(zrange*tRes, prange, y[:,:,i])
        Y[:,:,i] = f(t, prange)
    return points,t,np.squeeze(Y)

def run(temp):
    from utils import autodict,workingSpaceVaribles
    #global embryo
    globals().update(temp)

    print(phenotype+'-'+ID,end=' ')
    outline = io.imread(pjoin(pathlst['outline'],'poly-'+ID+'.tif')).astype(bool)
    outlinexy = np.where(outline)
    globals().update(autodict(outline, outlinexy))

    img,y,ys,ybc = movavg_bc(pjoin(pathlst['rawtif'],phenotype,ID),channel='G')
    IMG,IMGSM,IMGD = diff(img,6,9)
    print('diff',end=' ')
    imr = img.copy()
    if not LabelR=='None':
        imr,y,ys,ybc = movavg_bc(pjoin(pathlst['rawtif'],phenotype,ID),channel='R',dt=2.4,gauss=3)#
    
    from lib.read_roi import read_roi_zip
    mcircles = np.array([[p['position'],p['y'][0],p['x'][0]] for p in read_roi_zip(pjoin(pathlst['manual_circles'], phenotype+'-'+ID+'.zip')).values()])
    mcircles,tkinetics,mkinetics = kinetics(mcircles, IMGD, [-20,40], radius=7)
    threshlst = mkinetics[:,np.logical_and(tkinetics>-5,tkinetics<10)].max(1)
    THRESH = 0.22*np.percentile(threshlst,50)
    print('threshold',end=' ')

    IMGM = IMGD>THRESH
    onx,ony = np.where(~outline)
    IMGM[:,onx,ony] = False
    nzzs = np.where(IMGM.max(1).max(1))[0]
    EXC = excitation(IMGM)
    imshape = EXC.shape
    print('excitation',end=' ')

    from kinetics import candidates,cluster_center,ts_normalize
    from imagestack import new_excitation,cum
    NEWEXC = new_excitation(EXC,1)
    cumnewexc = cum(NEWEXC,int(7/tRes))
    points = candidates(cumnewexc)
    print(len(points),end=', ')
    points, acircles = cluster_center(points,EXC)
    acircles = acircles[acircles[:,0]*tRes<100]
    print(len(acircles),end=', ')

    acircles,tkinetics,akinetics = kinetics(acircles, np.stack([img,imr],3), [-30,60], radius=5)
    print('kinetics',end=' ')

    NCUT= int(round(150/tRes))
    newexcsum,frequency,wait = periodicity(NEWEXC[:NCUT],9,60)
    print('periodicity',end=' ')

    from spread import findContours,trajinfo
    cntdict,polydict,graph = findContours(EXC,tRes)
    graph,trajs,trajps,trajad = trajinfo(graph,cntdict,polydict,tRes)
    save_rois(pjoin(pathlst['trajs'],phenotype+'-'+ID+'.h5'),graph,cntdict)
    print('connect',end=' ')

    data = autodict(trajad, frequency, wait, tkinetics, akinetics)
    np.save(pjoin(pathlst['data'],phenotype+'-'+ID), data)
    
    locals().update(globals())
    ws = workingSpaceVaribles(locals())
    np.save(pjoin(pathlst['data'],phenotype+'-'+ID+'_ws'), ws)
    print('')
    return ws



