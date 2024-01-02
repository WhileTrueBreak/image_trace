import cv2
import numpy as np
import potrace
import matplotlib.pyplot as plt
import math
import sys
import imageFilter
import os
import random

# from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer
# import image_slicer
# from image_slicer import join
# from PIL import Image

def pathDist(points):
    pt = points[-1]
    td = 0
    for i in points:
        td += math.dist(pt, i)
        pt = i
    return td

def upscale(input_img, model_path='RealESRGAN_x4plus_anime_6B.pth'):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
    output, _ = upsampler.enhance(input_img, outscale=2)
    return output

def split(input_img, splits):
    sliced = np.split(input_img,splits,axis=0)
    blocks = [np.split(img_slice,splits,axis=1) for img_slice in sliced]
    return blocks

def upscale_slice(image, slice, model_path='RealESRGAN_x4plus_anime_6B.pth'):
    width, height, _ = image.shape
    tiles = split(image, slice)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
    c = 0
    for i, rows in enumerate(tiles):
        for j, tile in enumerate(rows):
            output, _ = upsampler.enhance(tile, outscale=4)
            tiles[i][j] = output
            c += 1
    for i in tiles:
        for j in i:
            print(j.shape)
    rows = [np.hstack(i) for i in tiles]
    print(rows[0].shape)
    img_complete = np.vstack(rows)
    print(img_complete.shape)
    return img_complete

def getPaths(img, nudge=0.33):
    print('reading image')
    img = cv2.imread(img)  # Read image

    # if(min(img.shape[0:2]) < 1000):
    #     img = upscale_slice(img, 5)
    #     print(img.shape)

    print('filtering image')
    filtered = cv2.bilateralFilter(img, 5, 50, 50)
    filtered = cv2.addWeighted(filtered, 1, filtered, 0, 1)
    # filtered = imageFilter.kClusterColors(filtered    # filtered = cv2.GaussianBlur(filtered, (0,0), sigmaX=2, sigmaY=2)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)

    median = max(10, min(245, np.median(gray)))
    lower = int(max(0, (1 - nudge) * median))
    upper = int(min(255, (1 + nudge) * median))

    # print(lower, upper)

    print('getting image edge')
    edged = cv2.Canny(gray, lower, upper)
    # edged = cv2.Canny(gray, 50, 150)

    # cv2.imshow('img', gray)
    # cv2.waitKey(0)

    print('tracing image')
    bmp = potrace.Bitmap(edged)
    return bmp.trace(turdsize=10), edged

def formatPaths(path, img, norm=(1,1)):
    print('formatting paths')
    shape = img.shape

    maxDim = max(shape[0:2])
    maxNorm = max(norm)
    ratio = maxNorm/maxDim

    mx = 0
    my = 0

    paths = []

    td = 0

    for i in range(len(path)):
        xp,yp = [],[]
        xp.append(round(path[i][-1].end_point.x*ratio,4))
        yp.append(round(path[i][-1].end_point.y*ratio,4))
        for j in path[i]:
            if j.is_corner:
                xp.append(round(j.c.x*ratio,4))
                yp.append(round(j.c.y*ratio,4))
            else:
                xp.append(round(j.c1.x*ratio,4))
                yp.append(round(j.c1.y*ratio,4))
                xp.append(round(j.c2.x*ratio,4))
                yp.append(round(j.c2.y*ratio,4))
            xp.append(round(j.end_point.x*ratio,4))
            yp.append(round(j.end_point.y*ratio,4))
        yp = [1-x for x in yp]
        a = [(xp[i], yp[i]) for i in range(len(xp))]
        paths.append(a)
        if pathDist(a) < max(shape)/100: continue
        td += pathDist(a)
        if max(xp) > mx:
            mx = max(xp)
        if max(yp) > my:
            my = max(yp)
    paths.sort(key=pathDist, reverse=True)
    return paths

def exportPathsJava(paths):
    print('exporting paths for java')
    arr = [exportPathJava(path) for path in paths]
    return f'Vector2D[][] paths = {{{",".join(arr)}}};'
    return arr

def exportPathJava(path):
    fArr = []
    for coord in path:
        fArr.append(f'new Vector2D({round(coord[0], 4)},{round(coord[1], 4)})')
    string = f'{{{",".join(fArr)}}}'
    return string

def exportPathsString(paths):
    print('exporting paths as string')
    arr = [exportPathString(path) for path in paths]
    return "|".join(arr)

def exportPathString(path):
    fArr = []
    for coord in path:
        fArr.append(f'{round(coord[0], 4)},{round(coord[1], 4)}')
    string = f'{"-".join(fArr)}'
    return string

def plotPaths(paths):
    fig = plt.figure()
    ax = fig.add_subplot()
    # pltimg = plt.imread(img_name)
    # ax.imshow(fimg)
    ax.set_aspect('equal', adjustable='box')

    paths.sort(key=pathDist, reverse=True)

    for path in paths:
        # print(path)
        xp = [i[0] for i in path]
        yp = [i[1] for i in path]
        plt.plot(xp, yp, 'k', linewidth=1)

    plt.show()

def cleanPaths(paths):
    print('cleaning paths')
    new = [cleanPath(i) for i in paths]
    print(f'Paths: {len(paths)}')
    print(f'Nodes: {sum(len(i)for i in paths)}')
    return new

def cleanPath(path, threshhold=0.0001):
    newPath = []
    last = None
    for i,e in enumerate(path):
        if last and (e[0]-last[0])**2+(e[1]-last[1])**2 < threshhold**2: continue
        newPath.append(e)
        last = e
    return newPath

def getRemoveSimilarPaths(paths, threshhold=0.001):
    print('removing overlap')
    toRemove = set()
    done = set()
    total = len(paths)*len(paths)
    tlength = len(str(total))
    barLength,_ = os.get_terminal_size()
    barLength -= tlength*2 + 3
    count = 0
    for i,p1 in enumerate(paths):
        for j,p2 in enumerate(paths):
            count += 1
            if p1 == p2: continue
            if i in toRemove: continue
            if (i,j) in done: continue
            if j in toRemove: continue

            p1B = getBounds(p1)
            p2B = getBounds(p2)

            if checkBounds(p1B, p2B, threshhold*2): continue
            if checkBounds(p2B, p1B, threshhold*2): done.add((j,i))

            print(f'|{"="*int(count/total*barLength)}{"_"*(barLength-int(count/total*barLength))}|{str(count).zfill(tlength)}/{total}', end='\r')
            dist, maxd = getPathAverageDist(p1, p2, threshhold=threshhold)
            if not dist: continue
            if dist < threshhold: toRemove.add(i)
    print(f'|{"="*barLength}|{str(count).zfill(tlength)}/{total}')
    print(f'removing {len(toRemove)} paths')
    for i in sorted(toRemove, reverse=True):
        del paths[i]
    
    print(f'Paths: {len(paths)}')
    print(f'Nodes: {sum(len(i)for i in paths)}')

def checkBounds(b1, b2, threshhold):
    if b1[0] >= b2[0]-threshhold and b1[1] >= b2[1]-threshhold and b1[2] <= b2[2]+threshhold and b1[3] <= b2[3]+threshhold: return False
    return True

def getBounds(path):
    x, y, X, Y = 1, 1, 0, 0
    for e in path:
        if e[0] < x: x = e[0]
        if e[1] < y: y = e[1]
        if e[0] > X: X = e[0]
        if e[1] > Y: Y = e[1]
    return x, y, X, Y

def getPathAverageDist(p1, p2, threshhold=0.001):
    maxd = 0
    p1d = 0
    p1l = len(p1)
    for i,e in enumerate(p1):
        dist = getDistFromPath(p2, e)
        if dist > threshhold*5: return None, dist
        if dist > maxd: maxd = dist
        p1d += dist
        if p1d/p1l > threshhold: return None, maxd
    p1d /= p1l
    return p1d, maxd

def getDistFromPath(path, point):
    last = path[-1]
    minDist = float('inf')
    for i,e in enumerate(path):
        dist = getDistFromLine(last, e, point)
        if dist < minDist: minDist = dist
        last = e
    return minDist;

def getDistFromLine(l1, l2, point):
    p = np.array(point)
    a = np.array(l1)
    b = np.array(l2)
    # Handle case where p is a single point, i.e. 1d array.
    p = np.atleast_2d(p)

    # TODO for you: consider implementing @Eskapp's suggestions
    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)[0]

def writeToFile(filename, text):
    with open(filename, "w") as file:
        file.write(text)

def main():
    img_name = sys.argv[1]
    path, fimg = getPaths(img_name)
    paths = formatPaths(path, fimg)
    paths = cleanPaths(paths)

    getRemoveSimilarPaths(paths, threshhold=0.005)

    txt_name = img_name.split('.')[0]+'_c'+'.txt'
    string = exportPathsString(paths)
    writeToFile(txt_name, string)
    plotPaths(paths)

if __name__ == '__main__':
    main()
