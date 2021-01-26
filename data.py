import numpy as np 
import os
import skimage.io as io

membranes = [255,0,0]
mitochondria = [0,255,0]
synapse = [0,0,255]
Unlabelled = [0,0,0]

def saveResult(save_path, npyfile, height, width):
    for i,item in enumerate(npyfile):
        img = np.zeros((height, width, 3))
        for k in range(height):
                for l in range(width):
                        if ((item[k][l][0] > item[k][l][1])and(item[k][l][0] > item[k][l][2])and(item[k][l][0] > item[k][l][3])):
                                img[k][l] = (0, 0, 0)
                        elif ((item[k][l][1] > item[k][l][0])and(item[k][l][1] > item[k][l][2])and(item[k][l][1] > item[k][l][3])):
                                img[k][l] = (255, 0, 0)
                        elif ((item[k][l][2] > item[k][l][0])and(item[k][l][2] > item[k][l][1])and(item[k][l][2] > item[k][l][3])):
                                img[k][l] = (0, 255, 0)
                        else:
                                img[k][l] = (0, 0, 255)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def saveResult2(save_path, npyfile, height, width):
    for i,item in enumerate(npyfile):
        img = np.zeros((height, width, 3))
        for k in range(height):
                for l in range(width):
                        if ((item[k][l][0] > item[k][l][1])and(item[k][l][0] > item[k][l][2])and(item[k][l][0] > item[k][l][3])):
                                img[k][l] = (0, 0, 0)
                        elif ((item[k][l][1] > item[k][l][0])and(item[k][l][1] > item[k][l][2])and(item[k][l][1] > item[k][l][3])):
                                img[k][l] = (255, 0, 0)
                        elif ((item[k][l][2] > item[k][l][0])and(item[k][l][2] > item[k][l][1])and(item[k][l][2] > item[k][l][3])):
                                img[k][l] = (0, 255, 0)
                        else:
                                img[k][l] = (0, 0, 255)
        io.imsave(os.path.join(save_path,"%d_predict2.png"%i),img)
