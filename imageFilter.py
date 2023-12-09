from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2

def kClusterColors(img):
    # img = cv2.imread(img)  # Read image

    img = cv2.GaussianBlur(img, (0,0), sigmaX=2, sigmaY=2)

    (h, w) = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = MiniBatchKMeans(n_clusters=7)
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w, 3))
    # img = img.reshape((h, w, 3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    # display the images and wait for a keypress
    # cv2.imshow("image", np.hstack([quant]))
    # cv2.waitKey(0)
    return quant

def reduceColors(img, colors=8):

    # Calculate the most common color in the image
    unique_colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    sorted_count_idx = np.argsort(counts)
    sorted_count = np.sort(counts)

    # get top n colors
    topNCountIdx = sorted_count_idx[-colors:]
    topNCount = counts[topNCountIdx]
    topNColors = unique_colors[topNCountIdx]

    # print(topNCount)
    # print(topNColors)

    majority_color = unique_colors[np.argmax(counts)]

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return img

def getIndex(arr, item):
    for idx, i in enumerate(arr):
        if (i==item).all(): return idx
    return -1

def getColorDiff(c1, c2):
    t = 0
    for i in range(3):
        t += abs(int(c1[i])-int(c2[i]))
    return t

def getNeighbours(imgArray, i, j):
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbours = []
    w, h, _ = imgArray.shape
    for offset in offsets:
        ni = i+offset[0]
        nj = j+offset[1]
        if ni < 0 or ni >= w: continue
        if nj < 0 or nj >= h: continue
        neighbours.append((ni, nj))
    return neighbours

# findColors('senjougahara.jpg')
# findColors('ba.png')
