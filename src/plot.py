'''
visualization
'''
import numpy as np
def mean_CI(data,minsample=2):
    '''
    mean and CI of many time series

    :param data: 2D array. each row is one timeseries. each column is measurements of mutiple samples at the same time point
    :return: 1d array of mean and CI
    '''
    from scipy.stats import sem
    NT = data.shape[1]# number of time points for each measurement
    mean = np.full(NT,np.nan)
    CI = np.full(NT,np.nan)
    for i in np.arange(NT):
        mask = ~np.isnan(data[:,i])
        if mask.sum()>=minsample:
            d = data[mask,i]
            mean[i] = np.nanmean(d)
            CI[i] = 1.96*sem(d)
    return mean,CI

def centroid(cnt):
    '''
    find centroid of opencv contour
    :return: tuple(cX,cY)
    '''
    import cv2
    M = cv2.moments(cnt)
    if M["m00"]==0:
        return None
    else:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return (cX,cY)

def colormap(v,vmin,vmax,cmap='jet'):
    '''
    convert number to rgb value using given colormap
    
    :param v: number to be converted
    :type v: number or numpy array
    :param vmin/vmax: bounds of colormap
    :type vmin/vmax: float
    :param cmap: colormap name in Matplotlib
    :type cmap: str
    :return: converted rgb array
    :rtype: np.uint8 array
    
    .. note:: to convert 3D(Z,X,Y) array, iterate in Z to convert 2D(X,Y)

    '''
    import matplotlib.cm
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    temp = (np.array(m.to_rgba(v))*255).astype(np.uint8)
    if isinstance(v, (int, float, np.int64, np.float32)):
        return temp[:3]
    elif len(v.shape)==1:
        return temp[:,:3]
    elif len(v.shape)==2:
        return temp[:,:,:3]


def crop_border(imstack,rect,value, top=2, bottom=2, left=2, right=2):
    '''
    crop and add border to imagestack
    
    :param imstack: np.array (Z,X,Y) or (Z,X,Y,3)
    :param rect: crop rectangle [[z0,z1],[x0,x1],[y0,y1]]
    :param value: int or RGB value to fill the border pixels
    :param top,bottom,left,right: width and height of the border
    '''
    import cv2
    imstack = imstack[rect[0][0]:rect[0][1],rect[2][0]:rect[2][1],rect[1][0]:rect[1][1]]
    imstack = np.stack([cv2.copyMakeBorder(imstack[pz], top, bottom, left, right, cv2.BORDER_CONSTANT,None, value) for pz in range(imstack.shape[0])],0)
    return imstack

def scalebar_timestamp(im,dt,fgc):# dt: time between frames, fgc: foreground color
    '''
    add space scale bar and time stamp on imagestack
    '''
    import cv2
    xp = im.shape[1]-10
    cv2.line(im[0], (5,xp), (25,xp), fgc, 1)# scale bar
    cv2.putText(im[0], '2um', (5,xp-5), cv2.FONT_HERSHEY_SIMPLEX , 0.5, fgc, 1)
    for i in range(im.shape[0]):
        cv2.putText(im[i], str(round(dt*i,1))+' S', (10,15), cv2.FONT_HERSHEY_SIMPLEX , 0.5, fgc, 1)


def color_excitation(traj,c,im,exc,newexc):
    '''
    color code different connected excitations
    
    :param traj: [(pz0,cnt0),(pz1,cnt1),...]
    :param c: RGB color pair [newexc_color, exc_color]
    :param im: RGB imagestack
    :param exc,newexc: binary imagestack
    '''
    from skimage.draw import polygon
    for pz,cnt in traj:
        fpx,fpy = polygon(cnt[:,0,1],cnt[:,0,0])# filled polygon index
        fpnz = np.where(exc[pz,fpx,fpy])[0]
        im[pz,fpx[fpnz],fpy[fpnz]] = c[1]
        fpnz = np.where(newexc[pz,fpx,fpy])[0]
        im[pz,fpx[fpnz],fpy[fpnz]] = c[0]


def draw_circles(imstack,point,color,N,rlst):
    '''
    draw circle perimeter on RGB image stack
    
    :param imstack: np.array((Z,X,Y,3), np.uint8)
    :param point: [z,x,y]
    :param color: [R,G,B]
    :param N: draw circle from frame z-N to frame z+N
    :param radius: list of radius of circles
    '''
    from skimage.draw import circle_perimeter
    pz,px,py = point
    radius = max(rlst)+1
    _,XMAX,YMAX,_ = imstack.shape
    if px<radius or py<radius or XMAX-px<radius or YMAX-py<radius:
        return
    if N>0:
        zrange = np.arange(max(0,pz-N),min(imstack.shape[0],pz+N+1))
    else:
        zrange = [pz]
    #print(zrange[0],zrange[-1],pz,px,py)
    for radius in rlst:
        rr,cc = circle_perimeter(px,py,radius)
        for zz in zrange:
            for channel in range(3):
                imstack[zz,rr,cc,channel] = color[channel]


def hSV(rgb,channel=2,v=155):
    '''
    given rgb value, change its saturation and value in hsv mode
    '''
    import matplotlib
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[channel] = v
    return matplotlib.colors.hsv_to_rgb(hsv).astype(np.uint8)

