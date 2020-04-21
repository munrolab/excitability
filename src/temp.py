
def connect(exc):
    '''
    find contour of blobs, connect overlapping blobs into trajectories
    :param exc: binary mask of excited regions
    :return: dict, each element is a list [(pz,cnt),(pz+1,cnt),...]
    '''
    globals().update(embryo)
    import cv2
    from spread import distance,DFS
    # detect contour of blobs
    graph = {}# key:(pz,i), stores index of connected blob
    cntdict = {i:[] for i in nzzs}# key:(pz,i), stores contours
    for pz in nzzs:
        cntdict[pz] = []
        contours, hierarchy = cv2.findContours(exc[pz].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i,cnt in enumerate(contours):
            cntdict[pz].append(cnt)
            graph[(pz,i)] = []

    # measure distance between blobs and save connections in graph
    DT = int(round(3/tRes))
    for pz in nzzs[:-1]:
        #print(pz, end =" ")
        for j,cnt in enumerate(cntdict[pz]):
            for pz2 in range(min(pz+1,nzzs[-1]),min(pz+DT,nzzs[-1])):
                found = False
                for j2,cnt2 in enumerate(cntdict[pz2]):
                    d = distance(cnt,cnt2)
                    if d>-2:
                        graph[(pz,j)].append((pz2,j2))# split only
                        graph[(pz2,j2)].append((pz,j))# merge only
                        found = True
                if found:
                    break

    # find trajectories based on blob-blob connections
    keys = list(graph.keys())
    visited = {k:False for k in keys}# Mark all the vertices as not visited 
    trajcnts = {}
    for k in keys:
        if visited[k]:
            continue
        else:
            traj = []#stores index of connected blobs
            DFS(graph, k, traj, visited)
            trajcnts[len(trajcnts)] = [(pz,cntdict[pz][i]) for pz,i in traj]#element: (pz,cnt)
    return trajcnts



def circles(imdiff, newexc, exc, tgroup=10, t0min=25):
    '''
    place circles in regions where new excitation starts
    
    :param newexc: binary imagestack of new excitation
    :param tgroup: persistence time of new excitation, used to calculated cumulative new excitation
    :param t0min: new excitation time is not accurate at the start of videos, so don't place circles in this time window
    '''
    global tRes, nzzs
    from skimage.morphology import disk,erosion,remove_small_objects
    from skimage.feature import peak_local_max
    from skimage.draw import circle
    from utils import binary_segments
    #from kinetics import cluster_center

    N = int(round(tgroup/tRes))
    points = []
    cumnewexc = np.zeros(newexc.shape,dtype=bool)
    for pz in nzzs[N+1:]:
        cumnewexc[pz] = newexc[pz-N:pz+1].max(0)# group mask in N frames
    imp = np.zeros(newexc.shape,dtype=float)# diff in eroded cum newexc mask
    imdiff[imdiff<0] = 0
    selem = disk(2)
    Z0MIN = int(round(t0min/tRes))
    _,XMAX,YMAX = newexc.shape
    radius=5
    for pz in np.arange(Z0MIN,nzzs[-1]+1):
    #for pz in [234]:
        temp = erosion(cumnewexc[pz], selem)
        imp[pz][temp] = imdiff[pz][temp]
        pxys = peak_local_max(imp[pz], min_distance=int(radius))
        for px,py in pxys:#[:5]
            if px<radius or py<radius or XMAX-px<radius or YMAX-py<radius:
                continue
            rr,cc = circle(px,py,radius)
            ts = exc[:,rr,cc].mean(1)
            segs = binary_segments(ts>0,onlytrue=True)
            if len(segs)==0:
                continue
            seg = segs[np.where(segs[:,1]<=pz)[0][-1]]
            pz0,pz1 = seg[1],seg[1]+seg[2]# z bounds for determining nearby circles to be of the same cluster
            if cumnewexc[pz0:pz1+1,rr,cc].mean(1).max()<0.75:
                continue
            
            ts = imdiff[pz0:pz1+1,rr,cc].mean(1)
            pzz = pz0+np.argmax(ts)-1# z position corrected for max of diff
            
            if pzz<Z0MIN:
                continue
            w = imdiff[pz0:pz1+1,rr,cc].mean(1).max()# weights for sorting lst of points
            points.append([pzz,px,py, pz0,pz1, w])
    points = np.array(points)
    points = points[np.argsort(-1*points[:,-1]),:-1].astype(int)
    points = cluster_center(points,D=10)
    return points

def cluster_center(points,D=5):
    '''
    [pz,px,py,pz0,pz1] -> [pzz,px,py,pz0,pz1,center_center,cluster_member]
    '''
    #find clusters of points and keep only points with hightest weights
    points = np.hstack([points,np.full((points.shape[0],2),-1)]).astype(int)
    # label whether a point is visited. not visited point is -1
    idxs = np.where(points[:,-1]==-1)[0]
    while len(idxs)>0:# stop when all points are visited
    #for i in range(2):
        isd = idxs[0]# index of seed point
        pz,px,py,pz0,pz1 = points[isd,:5]
        tooclose = points_distance(points,[px,py,pz0,pz1],D)
        tooclose_labeled = tooclose[points[:,-1]>-1]
        if len(tooclose_labeled)>0 and tooclose_labeled.max()==1:
            temp = np.logical_and(tooclose, points[:,-1]>-1)
            LABEL = points[temp][0][-1]
            points[isd,-1] = LABEL
            idxs = np.where(points[:,-1]==-1)[0]
            continue
        mask = np.logical_and(tooclose, points[:,-1]==-1)
        LABEL = points[:,-1].max()+1
        points[mask,-1] =  LABEL# label all points in cluster as visited
        points[isd,-2] = LABEL
        idxs = np.where(points[:,-1]==-1)[0]
    return points



def position(pzxy,EXC,imdiff,radius=5):
    pz,px,py = pzxy
    _,XMAX,YMAX = EXC.shape
    if px<radius or py<radius or XMAX-px<radius or YMAX-py<radius:
        return None
    rr,cc = circle(px,py,radius)
    ts = EXC[:,rr,cc].mean(1)
    segs = binary_segments(ts>0,onlytrue=True)
    if len(segs)==0:
        return None
    seg = segs[np.where(segs[:,1]<=pz)[0][-1]]
    pz0,pz1 = seg[1],seg[1]+seg[2]# z bounds for determining nearby circles to be of the same cluster
    if EXC[pz0:pz1+1,rr,cc].mean(1).max()<0.75:
        return None

    ts = imdiff[pz0:pz1+1,rr,cc].mean(1)
    pzz = pz0+np.argmax(ts)-1# z position corrected for max of diff
    return pz0,pz1,pzz





pz0 = 136
N = 5
pzlst = [pz0-10]+list(np.arange(pz0,pz0+15+1,N))
circles_byz = {}
for i in range(len(pzlst)-1):
    circles_byz[pzlst[i]] = []
    for p in acircles:
        if rect[2][0]<=p[1] and p[1]<=rect[2][1] and rect[1][0]<=p[2] and p[2]<=rect[1][1]:
            if pzlst[i]<=p[0] and p[0]<pzlst[i+1]:
                circles_byz[pzlst[i]].append(list(p)+['-'])
            if pzlst[i]<=p[0] and p[0]<pzlst[-1]:
                circles_byz[pzlst[i]].append(list(p)+['-'])
circles_byz



# -------------------- ts --------------------
loc = {'L':2.5,'W':2.4,'T':0.75,'H':6}
axes = fig.subplots(ncols=2, nrows=3, sharex='row', sharey='row',
                    gridspec_kw={'left':loc['L']/FW, 'right':(loc['L']+loc['W'])/FW, 
                                'top':1-loc['T']/FH,'bottom':1-(loc['T']+loc['H'])/FH,
                                'height_ratios':[3,1.5,1],
                                'wspace':0.4, 'hspace':0.5})


def ts_normalize(T,arr,window=[-20,0]):
    temp = np.logical_and(window[0]<=T, T<=window[1])
    idxes_min = np.zeros((arr.shape[0]),dtype=int)
    arr2 = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        idx_min = np.where(temp)[0][0]+np.argmin(arr[i,temp,0])
        idxes_min[i] = idx_min
        scale_factor = arr[i,idx_min,0]#:
        arr2[i] = arr[i]/scale_factor
    return idxes_min, arr2




def circle_mean_intensity(zxy, im, PRE,POST,MIN,MAX, radius=7):
    '''
    mean intensity of pixels in circle

    :param zxy: coordinate of circle center
    :type zxy: list(int)
    :param im: image stack
    :type im: np.array(float)
    :param PRE,POST: padding around cz
    :param MIN,MAX: absolute cutoffs in z where measurement is allowed
    :param radius: radius of circle
    :param radius: int
    '''
    from skimage.draw import circle
    cz,cx,cy = zxy
    zrange = np.arange(max(MIN,cz+PRE),min(MAX,cz+POST)).astype(int)
    z = zrange-cz
    rr,cc = circle(cx,cy,radius)
    y = im[zrange][:,rr,cc].mean(1)#average over NPixels
    return z,y

def extract_ts(t,y,T):# sample y against time T
    shape = list(y.shape)
    shape[0] = len(T)
    Y = np.full(shape,np.nan)
    mask = np.logical_and(T>=t[0],T<=t[-1])
    if len(y.shape)==1:
        Y[mask] = np.interp(T[mask],t,y)
    else:
        for i in range(y.shape[1]):
            Y[mask,i] = np.interp(T[mask],t,y[:,i])
    return Y

def kinetics(points,imhyperstack,T,radius=7):
    '''
    measure mean intensity in circles as a functin of time
    
    :return: np.array((Ncircles, time, channel))
    '''
    global tRes, nzzs
    from utils import circle_mean_intensity,extract_ts
    PRE,POST = round(T[0]/tRes),round((T[-1]+1)/tRes)
    MIN,MAX = nzzs[0],nzzs[-1]
    ylst = []
    for p in points:
        z,y = circle_mean_intensity(p, imhyperstack, PRE,POST,MIN,MAX, radius=radius)
        y = extract_ts(z*tRes, y, T)
        ylst.append(y)
    return np.stack(ylst,0)

def kinetics(points,imhyperstack,T,radius=7):
    '''
    measure mean intensity in circles as a functin of time
    
    :return: np.array((Ncircles, time, channel))
    '''
    global tRes, nzzs
    from utils import circle_mean_intensity,extract_ts
    PRE,POST = round(T[0]/tRes),round((T[-1]+1)/tRes)
    MIN,MAX = nzzs[0],nzzs[-1]
    ylst = []
    for p in points:
        z,y = circle_mean_intensity(p, imhyperstack, PRE,POST,MIN,MAX, radius=radius)
        y = extract_ts(z*tRes, y, T)
        ylst.append(y)
    return np.stack(ylst,0)
akinetics = kinetics(acircles, np.stack([img,imr2],3), tkinetics, radius=5)
idx_nonan = np.where(~np.isnan(akinetics).max(1).max(1))[0]
acircles,akinetics = acircles[idx_nonan],akinetics[idx_nonan]

        
        ts = imdiff[pz0:pz1+1,rr,cc].mean(1)
        points[isd,0] = pz0+np.argmax(ts)-1# z position corrected for max of diff

def circles(imdiff, newexc, exc, tgroup=10, t0min=25, radius=8):
    '''
    place circles in regions where new excitation starts
    
    :param newexc: binary imagestack of new excitation
    :param tgroup: persistence time of new excitation, used to calculated cumulative new excitation
    :param t0min: new excitation time is not accurate at the start of videos, so don't place circles in this time window
    '''
    global tRes, nzzs
    from skimage.morphology import disk,erosion,remove_small_objects
    from skimage.feature import peak_local_max
    from skimage.draw import circle
    from utils import binary_segments
    from kinetics import cluster_center

    N = int(round(tgroup/tRes))
    points = []
    cumnewexc = np.zeros(newexc.shape,dtype=bool)
    for pz in nzzs[N+1:]:
        cumnewexc[pz] = newexc[pz-N:pz+1].max(0)# group mask in N frames
    imp = np.zeros(newexc.shape,dtype=float)# diff in eroded cum newexc mask
    imdiff[imdiff<0] = 0
    selem = disk(2)
    Z0MIN = int(round(t0min/tRes))
    _,XMAX,YMAX = newexc.shape
    for pz in np.arange(Z0MIN,nzzs[-1]+1):
    #for pz in [234]:
        temp = erosion(cumnewexc[pz], selem)
        imp[pz][temp] = imdiff[pz][temp]
        pxys = peak_local_max(imp[pz], min_distance=int(radius))
        for px,py in pxys:#[:5]
            if px<radius or py<radius or XMAX-px<radius or YMAX-py<radius:
                continue
            rr,cc = circle(px,py,radius)
            ts = exc[:,rr,cc].mean(1)
            segs = binary_segments(ts>0,onlytrue=True)
            if len(segs)==0:
                continue
            seg = segs[np.where(segs[:,1]<=pz)[0][-1]]
            pz0,pz1 = seg[1],seg[1]+seg[2]# z bounds for determining nearby circles to be of the same cluster
            
            rr,cc = circle(px,py,4)
            ts = cumnewexc[pz0:pz1+1,rr,cc].mean(1)
            if ts.max()<0.9:
                continue
            pzz = pz0+np.where(ts>0.1)[0][0]# z position corrected for beginning of excitation
            
            if pzz<Z0MIN:
                continue
            
            w = imdiff[pz0:pz1+1,rr,cc].mean(1).max()# weights for sorting lst of points
            points.append([pzz,px,py, pz0,pz1, w])
    points = np.array(points)
    points = points[np.argsort(-1*points[:,-1]),:-1].astype(int)
    points = cluster_center(points)
    return points

