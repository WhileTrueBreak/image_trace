import cv2
import numpy as np
import potrace
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import math
import sys

Path = mpath.Path

def pathDist(points):
    pt = points[-1]
    td = 0
    for i in points:
        td += math.dist(pt, i)
        pt = i
    return td

t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold

img = cv2.imread(sys.argv[1])  # Read image
edge = cv2.Canny(img, t_lower, t_upper)
bmp = potrace.Bitmap(edge)
path = bmp.trace()

img_h = img.shape[0]

fig = plt.figure()
ax = fig.add_subplot()
for i in range(len(path)):
    last_ep = path[i][-1].end_point
    for seg in path[i]: 
        if seg.is_corner:
            plt.plot([last_ep.x, seg.c.x, seg.end_point.x], [img_h - last_ep.y, img_h - seg.c.y, img_h - seg.end_point.y], 'k', linewidth=1)
        else:
            plt.plot([last_ep.x, seg.c1.x], [img_h - last_ep.y, img_h - seg.c1.y], 'k', linewidth=1)
            pp = mpatches.PathPatch(
                Path([ (seg.c1.x, img_h - seg.c1.y), (seg.c2.x, img_h - seg.c2.y), (seg.end_point.x, img_h - seg.end_point.y)],
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                fc="none", transform=ax.transData)
            ax.add_patch(pp)
        last_ep = seg.end_point
    # xp,yp = [],[]
    # xp.append(path[i][-1].end_point.x)
    # yp.append(img_h - path[i][-1].end_point.y)
    # for j in path[i]:
    #     if j.is_corner:
    #         xp.append(j.c.x)
    #         yp.append(img_h - j.c.y)
    #     xp.append(j.end_point.x)
    #     yp.append(img_h - j.end_point.y)
    # paths.append((xp, yp))
    # a = [(xp[i], yp[i]) for i in range(len(xp))]
    # print(pathDist(a))
    # if pathDist(a) < 1: continue
    # plt.plot(xp, yp)
    # if max(xp) > mx:
    #     mx = max(xp)
    # if max(yp) > my:
    #     my = max(yp)

ax.set_aspect('equal', adjustable='box')
plt.show()

