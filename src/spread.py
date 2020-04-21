# coding: utf-8
# Created on: 01.08.2016
# Author: Baixue Yao
"""
spread of excitation

| trajectory of contours has 2 formats: 
| traj ``[(pz0,cnt0), (pz1,cnt1), ...]`` 
| traj_z  ``{pz: [cnt0, cnt1,...], pz+1: [cnt2, cnt3, ...], ...}``
"""

import numpy as np

def findContours(exc,tRes):
    '''
    detect contour of blobs; connect overlapping blobs into trajectories

    :return: dict, each element is a list [(pz,cnt),(pz+1,cnt),...]
    '''
    import cv2
    from skimage.draw import polygon
    #exc = exc[:N]#.copy()
    #exc[:,~outline] = False

    nzzs = np.where(exc.max(1).max(1))[0]
    cntdict = {i:[] for i in nzzs}# key:(pz,i), stores contours
    polydict = {i:[] for i in nzzs}
    for pz in nzzs:
        cntdict[pz] = []
        contours, hierarchy = cv2.findContours(exc[pz].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>100:
                cntdict[pz].append(cnt)
                poly = [(x,y) for x,y in np.stack(polygon(cnt[:,0,1],cnt[:,0,0]),1)]
                polydict[pz].append(poly)

    graph = {}# key:(pz,i), stores index of connected blob
    for pz in cntdict.keys():
        for i in range(len(cntdict[pz])):
            graph[(pz,i)] = [[],[],-1]

    # measure distance between blobs and save connections in graph
    DT = int(round(3/tRes))
    nzzs = list(cntdict.keys())
    for pz in nzzs[:-1]:
        #print(pz, end =" ")
        for j,cnt in enumerate(cntdict[pz]):
            for pz2 in range(min(pz+1,nzzs[-1]),min(pz+DT,nzzs[-1])):
                found = False
                for j2,cnt2 in enumerate(cntdict[pz2]):
                    if distance(cnt,cnt2)==0:
                        fp,fp2 = polydict[pz][j],polydict[pz2][j2]
                        n = len(set(fp).intersection(fp2))
                        w,w2 = round(n/len(fp2),2),round(n/len(fp),2)
                        graph[(pz2,j2)][0].append((w,(pz,j)))# parents
                        graph[(pz,j)][1].append((w2,(pz2,j2)))# chidren
                        found = True
                if found:
                    break
    return cntdict,polydict,graph

def distance(cnt,cnt2):
    '''
    distance between two contours

    | if overlap, distance = ``0`` 
    | otherwise, distance < ``0``

    :param cnt,cnt2: contour of blob in opencv format
    :type cnt,cnt2: np.array(int)
    :return: minimum distance between two contours
    :rtype: float
    '''
    import cv2
    dlst = []
    for p in cnt[:,0,:]:
        d = cv2.pointPolygonTest(cnt2,tuple(p),True)#positive (inside), negative (outside)
        if d>=0:
            return 0
        dlst.append(d)
    
    for p in cnt2[:,0,:]:
        d = cv2.pointPolygonTest(cnt,tuple(p),True)#positive (inside), negative (outside)
        if d>=0:
            return 0
        dlst.append(d)
    return max(dlst)# since all elements in dlst are negative, max is minimum distance

def DFS(graph, v, LABEL): 
    '''
    Depth First Search(Traversal) for a graph

    :param v: vertex in graph
    '''
    graph[v][2] = LABEL# Mark the current node as visited  
    for w,v2 in graph[v][0]+graph[v][1]: # Recur for all the vertices adjacent to this vertex
        if graph[v2][2]<0: # graphs may contain cycles, so we may come to the same node again. To avoid processing a node more than once, we use a boolean visited array
            DFS(graph, v2, LABEL)


def trajinfo(graph,cntdict,polydict,tRes):
    LABEL = 0
    for v in graph.keys():#[keys[2]]:
        if graph[v][2]<0:
            DFS(graph, v, LABEL)
            LABEL += 1

    trajs, trajps = {}, {}
    for (pz,i),(parents,children,LABEL) in graph.items():
        cnt = cntdict[pz][i]
        p = polydict[pz][i]
        if LABEL not in trajs.keys():
            trajs[LABEL] = {}
            trajps[LABEL] = {}
        if pz not in trajs[LABEL].keys():
            trajs[LABEL][pz] = [cnt]
            trajps[LABEL][pz] = [p]
        else:
            trajs[LABEL][pz].append(cnt)
            trajps[LABEL][pz].append(p)

    trajad = {}
    for tID in trajps.keys():
        trajp = trajps[tID]
        p = []
        for pp in trajp.values():
            for ppp in pp:
                p += ppp
        a = len(set(p))/100.0
        pzs = list(trajp.keys())
        d = (max(pzs)-min(pzs))*tRes
        trajad[tID] = [a,d]
    return graph,trajs,trajps,trajad
    
def area_distr(trajad):
    ad = np.array(list(trajad.values()))
    alst,dlst = ad[:,0],ad[:,1]#list of total area covered by traj
    abins = list(np.arange(1,10,1))+list(np.arange(10,100,10))+list(np.arange(100,1000,50))+list(np.arange(1000,3000,100))
    asums = np.zeros(len(abins))
    idxs = np.digitize(alst, abins)
    for i in range(len(alst)):
        asums[idxs[i]]+=alst[i]
    #asums = np.cumsum(asums/asums.sum())
    #ax.set(ylabel='Fraction',ylim=[0,1.05],yticks=[0,0.5,1])
    return alst,dlst,abins,asums

def concave_hull(traj):
    '''
    find concave hull of traj

    :return: ``shapely.geometry.polygon.Polygon`` or ``shapely.geometry.multipolygon.MultiPolygon``
    '''
    from skimage.draw import polygon
    import alphashape
    points = np.fliplr(np.vstack([np.vstack(polygon(cnt[:,0,1],cnt[:,0,0])).T for pz,cnt in traj]))#
    return alphashape.alphashape(points, 1.0)
