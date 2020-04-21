# coding: utf-8
# Created on: 08.01.2016
# Author: Roman Miroshnychenko aka Roman V.M.
# E-mail: romanvm@yandex.ua
"""
miscellaneous

more
"""

'''

'''
import numpy as np

def rawtiff(path,channels):
    """
    get names of tiff videos from given path. Last character of filename is channel.

    :param path: a directory containing tiff images
    :type path: str
    :param channels: ['G'] for reading green channel only, ['G','R'] for reading both green and red channels
    :return: if channels=['G','R'], returns {'G':nameG, 'R':nameR}. if channels=['G'], returns nameG.
    """
    import os
    ID = path.split('/')[-1]
    namelist = os.listdir(path)#names of all file in directory
    #print(namelist)
    names = []
    for channel in channels:
        for name in namelist:
            if name.startswith(ID) and name.endswith(channel+'.tif'):
                names.append(path+'/'+name)# add file name if found a match
    if len(channels)==1:
        return names[0]
    return names


def randompxl(outlinexy, n=10):
    '''
    select n random pixels from binary mask outlinexy
    
    :return: list of pixels [[x0,y0],[x1,y1],...]
    '''
    import random
    otlnx,otlny = outlinexy
    idxrdn = random.sample(list(np.arange(0,len(otlnx),10)),n)
    return np.vstack([otlnx[idxrdn],otlny[idxrdn]]).T.tolist()

def crop(img,cnt):
    '''
    find min area bounding box of opencv contour and crop the rotated box
    '''
    import cv2
    rect = cv2.minAreaRect(cnt)
    #print("rect: {}".format(rect))
    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def binary_segments(y,onlytrue=False):
    '''
    divide 1D binary array into segments 

    :return: list of list. Segments are measured from left to right. Every inner list has 3 elements: [value(True or False), starting index, length]
    '''
    ydif = np.diff(y.astype(float))
    split_idx = np.sort(np.hstack((np.where(ydif>0)[0],np.where(ydif<0)[0])))+1
    segments = np.split(np.arange(len(y)),split_idx)
    if onlytrue:
        return np.array([(bool(y[s[0]]),s[0],len(s))for s in segments if bool(y[s[0]])])
    else:
        return np.array([(bool(y[s[0]]),s[0],len(s))for s in segments])


def autodict(*args):
    import inspect
    get_rid_of = ['autodict(', ',', ')', '\n']
    calling_code = inspect.getouterframes(inspect.currentframe())[1][4][0]
    calling_code = calling_code[calling_code.index('autodict'):]
    for garbage in get_rid_of:
        calling_code = calling_code.replace(garbage, '')
    var_names, var_values = calling_code.split(), args
    dyn_dict = {var_name: var_value for var_name, var_value in
                zip(var_names, var_values)}
    return dyn_dict

def workingSpaceVaribles(variable_dict):
    from types import ModuleType, FunctionType
    return {name:item for name,item in variable_dict.items() 
         if not name.startswith('_') 
         and not isinstance(item, (ModuleType,FunctionType))}

