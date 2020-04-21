import numpy as np
def candidates(imstack):
    imstack = imstack.astype(np.uint8)
    import cv2
    points = []
    nzzs = np.where(imstack.max(1).max(1))[0]
    for pz in nzzs:
        dist = cv2.distanceTransform(imstack[pz], cv2.DIST_L2, 3)
        idxx,idxy = np.where(dist>4)
        idxz = np.full(idxx.shape[0],pz)
        w = dist[idxx,idxy]
        points += np.stack([idxz,idxx,idxy,w],1).tolist()
    points = np.array(points)
    points = points[np.argsort(-1*points[:,-1]),:-1].astype(int)
    return points

def zbounds(ts,pz):
    from utils import binary_segments
    segs = binary_segments(ts,onlytrue=True)
    seg = segs[np.where(segs[:,1]<=pz)[0][-1]]
    pz0,pz1 = seg[1],seg[1]+seg[2]# z bounds for determining nearby circles to be of the same cluster
    return pz0,pz1

def cluster_center(points,exc,R=5):
    '''
    [pz,px,py,pz0,pz1] -> [pzz,px,py,pz0,pz1,center_center,cluster_member]
    '''
    from skimage.draw import circle
    rr,cc = circle(0,0,R)
    _,XMAX,YMAX = exc.shape
    #find clusters of points and keep only points with hightest weights
    points = np.hstack([points,np.zeros((points.shape[0],2))]).astype(int)
    # label whether a point is visited. not visited point is 0
    idxs = np.where(points[:,-1]==0)[0]
    counter = 0
    while len(idxs)>0 and counter<10**5:# stop when all points are visited
        counter += 1
        #print(len(idxs),end=', ')
        if len(idxs)==0:
            break
        isd = idxs[0]# index of seed point
        pz,px,py = points[isd,:3]
        
        if px<R or py<R or XMAX-px<R or YMAX-py<R:
            points[isd,-1] =  -1
            idxs = np.where(points[:,-1]==0)[0]
            continue
        
        pz0,pz1 = zbounds(exc[:,rr+px,cc+py].mean(1)>0, pz)
        mz = np.logical_and(points[:,0]>=pz0,points[:,0]<=pz1)
        dxy = np.sqrt((points[:,1]-px)**2+(points[:,2]-py)**2)
        
        LABEL = np.nanmax(points[:,-1])+1
        mask = np.logical_and(np.logical_and(mz,dxy<=10), points[:,-1]==0)
        points[mask,-1] =  LABEL# label all points in cluster as visited
        points[isd,-2] = LABEL
        idxs = np.where(points[:,-1]==0)[0]
        
        pz2,pz3 = zbounds(exc[pz0:pz1,rr+px,cc+py].mean(1)>0.5, pz)
        points[isd,0] = pz0+pz2# z position corrected for max of diff

    acircles = points[points[:,-2]>0,:3]
    return points, acircles


def ts_normalize(T,arr,window=[-7,-2]):
    temp = np.logical_and(window[0]<=T, T<=window[1])
    scale_factor = arr[:,temp].mean(1)#np.percentile(temp, 50,axis=1)
    scale_factor = np.expand_dims(scale_factor,axis=1)
    return scale_factor


def amp_distr(arr,tkinetics,MIN=0.5,MAX=2.5):
    mask = np.logical_and(0<=tkinetics, tkinetics<=20)
    arr = arr[:,mask].max(1)
    arr[arr<=MIN] = MIN
    arr[arr>=MAX] = MAX
    result,edges = np.histogram(arr, bins=np.arange(MIN,MAX+0.02,0.02), density=True)
    return arr,(edges[1:], result*(edges[1]-edges[0]))

