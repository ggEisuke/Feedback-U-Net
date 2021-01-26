from keras.models import *
from model import *
from data import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
from keras.utils.vis_utils import plot_model
import keras.callbacks
import os
from sklearn.utils.class_weight import compute_class_weight

#variable declaration
epochs = 1000
N_train = 192
N_val = 48
batch_size = 16
height = 256
width = 256
classes = 4
gpu = "1"

#GPU
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=gpu,
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

#make directory
if not os.path.exists("models"):
	os.mkdir("models")
if not os.path.exists("results"):
	os.mkdir("results")

#dataset
x_train=np.zeros((N_train,height,width,classes)).astype(np.float32)
y_train=np.zeros((N_train,height,width,classes)).astype(np.int32)
x_val=np.zeros((N_val,height,width,classes)).astype(np.float32)
y_val=np.zeros((N_val,height,width,classes)).astype(np.int32)

#loading train data
for k in range(0, N_train):
        print("train%d"%(k))
        im = load_img('dataset/Image/mCherry%04d.png' % (k), grayscale = True, target_size = (height, width), interpolation = "bilinear")
        im = img_to_array(im)
        im = im.astype(np.float32)
        x_train[k] = im.reshape(1, height, width, 1)
        for l in range(0,height):
            for m in range(0,width):
                    x_train[k][l][m][1] = x_train[k][l][m][0]
                    x_train[k][l][m][2] = x_train[k][l][m][0]
                    x_train[k][l][m][3] = x_train[k][l][m][0]

        im = load_img('dataset/GroundTruth/mCherry%04d.png' % (k), grayscale = False, target_size = (height,width))
        im = img_to_array(im)
        for i in range(0,height):
            for j in range(0,width):
                    if im[i][j][0] == 0 and im[i][j][1] == 0 and im[i][j][2] == 0:
                            y_train[k][i][j] = (1, 0, 0, 0)
                    elif im[i][j][0] == 255 and im[i][j][1] == 0 and im[i][j][2] == 0:
                            y_train[k][i][j] = (0, 1, 0, 0)
                    elif im[i][j][0] == 0 and im[i][j][1] == 255 and im[i][j][2] == 0:
                            y_train[k][i][j] = (0, 0, 1, 0)
                    else:
                            y_train[k][i][j] = (0, 0, 0, 1)

#loading validation data
for k in range(0, N_val):
        print("val%d"%(k))
        im = load_img('dataset/Image/mCherry%04d.png' % (k+N_train), grayscale = True, target_size = (height,width), interpolation = "bilinear")
        im = img_to_array(im)
        im = im.astype(np.float32)
        x_val[k] = im.reshape(1, height, width, 1)
        for l in range(0,height):
            for m in range(0,width):
                    x_val[k][l][m][1] = x_val[k][l][m][0]
                    x_val[k][l][m][2] = x_val[k][l][m][0]
                    x_val[k][l][m][3] = x_val[k][l][m][0]

        im = load_img('dataset/GroundTruth/mCherry%04d.png' % (k+N_train), grayscale = False, target_size = (height,width))
        im = img_to_array(im)
        for i in range(0,height):
            for j in range(0,width):
                    if im[i][j][0] == 0 and im[i][j][1] == 0 and im[i][j][2] == 0:
                            y_val[k][i][j] = (1, 0, 0, 0)
                    elif im[i][j][0] == 255 and im[i][j][1] == 0 and im[i][j][2] == 0:
                            y_val[k][i][j] = (0, 1, 0, 0)
                    elif im[i][j][0] == 0 and im[i][j][1] == 255 and im[i][j][2] == 0:
                            y_val[k][i][j] = (0, 0, 1, 0)
                    else:
                            y_val[k][i][j] = (0, 0, 0, 1)

#normalization
x_train /= 255.0
x_val /= 255.0

#compute class weight
y_integers = np.argmax(y_train, axis = 3)
y_reshape = y_integers.flatten()
class_weights = compute_class_weight('balanced', np.unique(y_reshape), y_reshape)

#defin model
model = unet(batch_size, height, width, classes)
model_checkpoint = ModelCheckpoint(filepath = os.path.join('models', 'model_best.hdf5'),
                                monitor = 'val_conv2d_10_mean_iou',
                                verbose = 1,
                                save_best_only = True,
                                mode = 'max',
                                save_weights_only = True)
"""
model_checkpoint = ModelCheckpoint(filepath = os.path.join('models', 'model_{epoch:02d}_{val_conv2d_10_mean_iou:.6f}.hdf5'),
                                monitor='val_conv2d_10_mean_iou',
                                verbose=1,
                                save_best_only = True,
                                mode = 'max',
                                save_weights_only = True)
"""

hist = model.fit(x_train,
                [y_train, y_train],
                batch_size = batch_size,
                epochs = epochs,
                verbose = 1,
                validation_data = (x_val, [y_val,y_val]),
                class_weight = [class_weights, class_weights],
                callbacks = [model_checkpoint])
model.summary()
plot_model(model, to_file='results/model.png')

#plot accuracy
def plot_history_meaniou(history):
        fig = plt.figure(1)
        plt.plot(hist.history['val_subtract_1_mean_iou'],label='first mean_iou')
        plt.plot(hist.history['val_conv2d_10_mean_iou'],label='second mean_iou')
        plt.xlabel('epoch')
        plt.ylabel('mean_iou')
        plt.legend()
        plt.savefig('results/mean_iou.png')
plot_history_meaniou(hist)

