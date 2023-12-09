import cv2
vidcap = cv2.VideoCapture('badapple.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 30
