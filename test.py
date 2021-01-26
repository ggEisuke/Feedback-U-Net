from keras.models import *
from model import *
from data import *
from keras.preprocessing.image import load_img, img_to_array
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np

#variable declaration
N_test = 80
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
if not os.path.exists("results/save"):
	os.mkdir("results/save")

PATH = "{}/test_accuracy.txt".format("results")
with open(PATH, mode = 'w') as f:
	pass

#dataset
x_test=np.zeros((N_test,height,width,classes)).astype(np.float32)
y_test=np.zeros((N_test,height,width,classes)).astype(np.int32)

#loading test data
for k in range(0, N_test):
        print("test%d"%(k))
        im = load_img('dataset/Image/mCherry%04d.png' % (k+240), grayscale = True, target_size = (height, width), interpolation = "bilinear")
        im = img_to_array(im)
        im = im.astype(np.float32)
        x_test[k] = im.reshape(1, height, width, 1)
        for l in range(0,height):
            for m in range(0,width):
                    x_test[k][l][m][1] = x_test[k][l][m][0]
                    x_test[k][l][m][2] = x_test[k][l][m][0]
                    x_test[k][l][m][3] = x_test[k][l][m][0]


        im = load_img('dataset/GroundTruth/mCherry%04d.png' % (k+240), grayscale = False, target_size = (height, width))
        im = img_to_array(im)
        for i in range(0,height):
            for j in range(0,width):
                    if im[i][j][0] == 0 and im[i][j][1] == 0 and im[i][j][2] == 0:
                            y_test[k][i][j] = (1, 0, 0, 0)
                    elif im[i][j][0] == 255 and im[i][j][1] == 0 and im[i][j][2] == 0:
                            y_test[k][i][j] = (0, 1, 0, 0)
                    elif im[i][j][0] == 0 and im[i][j][1] == 255 and im[i][j][2] == 0:
                            y_test[k][i][j] = (0, 0, 1, 0)
                    else:
                            y_test[k][i][j] = (0, 0, 0, 1)

#normalization
x_test /= 255.0

#defin model
model = unet(batch_size, height, width, classes)

#load weight
model.load_weights("models/model_best.hdf5")

#accucary
score = model.evaluate(x_test, [y_test, y_test], batch_size=batch_size, verbose=0)
with open(PATH, mode = 'a') as f:
        f.write("first round\n")
        f.write("meanIoU\t%f\n" % (score[3]))
        f.write("second round\n")
        f.write("meanIoU\t%f\n" % (score[4]))

#save output image
results, result2 = model.predict(x_test, batch_size = batch_size)
saveResult("results/save", results, height, width)
saveResult2("results/save", result2, height, width)