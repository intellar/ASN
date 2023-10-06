import cv2
import matplotlib.pyplot as plt
import numpy as np
import ASN_detector as ASN


#copyright intellar@intellar.ca

img = cv2.imread("./raw_camera_0_index_0.png",cv2.IMREAD_GRAYSCALE)
pts, accumulateur = ASN.ASN_detector(img)

#display
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
cimg[:,:,0] += np.array(accumulateur,dtype=np.uint8)
fig = plt.figure()
plt.imshow(cimg)
plt.plot(pts[:,0],pts[:,1],'g+', markersize=20, markeredgewidth=4)
plt.show()
