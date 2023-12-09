import cv2
import numpy as np
import potrace
import matplotlib.pyplot as plt
import math
import sys
import imageFilter

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

def formatPathsJava(paths):
    print('formatting paths for java')
    arr = [formatPathJava(path) for path in paths]
    return f'Vector2D[][] paths = {{{",".join(arr)}}};'
    return arr

def formatPathJava(path):
    fArr = []
    for coord in path:
        fArr.append(f'new Vector2D({round(coord[0], 4)},{round(coord[1], 4)})')
    string = f'{{{",".join(fArr)}}}'
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

def main():
    img_name = sys.argv[1]
    path, fimg = getPaths(img_name)
    paths = formatPaths(path, fimg)

    txt_name = img_name.split('.')[0]+'.txt'
    with open(txt_name, "w") as file:
        string = formatPathsJava(paths)
        file.write(string)
    plotPaths(paths)

if __name__ == '__main__':
    main()
