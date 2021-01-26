from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as keras

def iou_brack(y_iou):
    tp = keras.sum(keras.cast(keras.equal(y_iou, 7), keras.floatx()))
    f = keras.sum(keras.cast(keras.equal(y_iou, 8), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 9), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 10), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 14), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 21), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 28), keras.floatx()))
    return tp / (tp + f)

def iou_red(y_iou):
    tp = keras.sum(keras.cast(keras.equal(y_iou, 16), keras.floatx()))
    f = keras.sum(keras.cast(keras.equal(y_iou, 8), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 24), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 32), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 14), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 18), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 20), keras.floatx()))
    return tp / (tp + f)

def iou_green(y_iou):
    tp = keras.sum(keras.cast(keras.equal(y_iou, 27), keras.floatx()))
    f = keras.sum(keras.cast(keras.equal(y_iou, 9), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 18), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 36), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 21), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 24), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 30), keras.floatx()))
    return tp / (tp + f)

def iou_blue(y_iou):
    tp = keras.sum(keras.cast(keras.equal(y_iou, 40), keras.floatx()))
    f = keras.sum(keras.cast(keras.equal(y_iou, 10), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 20), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 30), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 28), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 32), keras.floatx())) + keras.sum(keras.cast(keras.equal(y_iou, 36), keras.floatx()))
    return tp / (tp + f) 

def mean_iou(y_true, y_pred):
    y_true2 = keras.argmax(y_true, axis = 3) + 1
    y_pred2 = keras.argmax(y_pred, axis = 3) + 7
    y_iou = y_true2* y_pred2

    brack = iou_brack(y_iou)
    red = iou_red(y_iou)
    green = iou_green(y_iou)
    blue = iou_blue(y_iou)
    return (brack + red + green + blue) / 4


def unet(batch_size, height, width, classes):

    convlstm11 = ConvLSTM2D(filters = 8, kernel_size = (3, 3), padding = 'same')
    convlstm12 = ConvLSTM2D(filters = 8, kernel_size = (3, 3), padding = 'same')

    convlstm21 = ConvLSTM2D(filters = 16, kernel_size = (3, 3), padding = 'same')
    convlstm22 = ConvLSTM2D(filters = 16, kernel_size = (3, 3), padding = 'same')

    conv31 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')
    conv32 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')

    conv41 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')
    conv42 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')

    convlstm51 = ConvLSTM2D(filters = 128, kernel_size = (3, 3), padding = 'same')
    convlstm52 = ConvLSTM2D(filters = 128, kernel_size = (3, 3), padding = 'same')

    deconv1 = Deconv2D(64, 2, strides = (2, 2), padding = "valid")
    conv61 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')
    conv62 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')

    deconv2 = Deconv2D(32, 2, strides = (2, 2), padding = "valid")
    conv71 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')
    conv72 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')

    deconv3 = Deconv2D(16, 2, strides = (2, 2), padding = "valid")
    convlstm81 = ConvLSTM2D(filters = 16, kernel_size = (3, 3), padding = 'same')
    convlstm82 = ConvLSTM2D(filters = 16, kernel_size = (3, 3), padding = 'same')

    deconv4 = Deconv2D(8, 2, strides = (2, 2), padding = "valid")
    conv93 = Conv2D(8, 3, padding = 'same', kernel_initializer = 'he_normal')
    convlstm91 = ConvLSTM2D(filters = 8, kernel_size = (3, 3), padding = 'same')
    convlstm92 = ConvLSTM2D(filters = 8, kernel_size = (3, 3), padding = 'same')

    conv10 = Conv2D(classes, 1, activation = 'softmax')

    inputs = Input(batch_shape=(batch_size,) + (height, width, classes))
    
    #####first round#####

    expand11 = Lambda(lambda x: keras.expand_dims(inputs, axis = 1))(inputs)
    mid11 = convlstm11(expand11)
    batch11 = BatchNormalization()(mid11)
    relu11 = Activation("relu")(batch11)
    expand12 = Lambda(lambda x: keras.expand_dims(relu11, axis = 1))(relu11)
    mid12 = convlstm12(expand12)
    batch12 = BatchNormalization()(mid12)
    relu12 = Activation("relu")(batch12)
    pool1 = MaxPooling2D(pool_size=(2, 2))(relu12)


    expand21 = Lambda(lambda x: keras.expand_dims(pool1, axis = 1))(pool1)
    mid21 = convlstm21(expand21)
    batch21 = BatchNormalization()(mid21)
    relu21 = Activation("relu")(batch21)
    expand22 = Lambda(lambda x: keras.expand_dims(relu21, axis = 1))(relu21)
    mid22 = convlstm22(expand22)
    batch22 = BatchNormalization()(mid22)
    relu22 = Activation("relu")(batch22)
    pool2 = MaxPooling2D(pool_size=(2, 2))(relu22)


    mid31 = conv31(pool2)
    batch31 = BatchNormalization()(mid31)
    relu31 = Activation("relu")(batch31)
    mid32 = conv32(relu31)
    batch32 = BatchNormalization()(mid32)
    relu32 = Activation("relu")(batch32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(relu32)


    mid41 = conv41(pool3)
    batch41 = BatchNormalization()(mid41)
    relu41 = Activation("relu")(batch41)
    mid42 = conv42(relu41)
    batch42 = BatchNormalization()(mid42)
    relu42 = Activation("relu")(batch42)
    drop4 = Dropout(0.5)(relu42)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


    expand51 = Lambda(lambda x: keras.expand_dims(pool4, axis = 1))(pool4)
    mid51 = convlstm51(expand51)
    batch51 = BatchNormalization()(mid51)
    relu51 = Activation("relu")(batch51)
    expand52 = Lambda(lambda x: keras.expand_dims(relu51, axis = 1))(relu51)
    mid52 = convlstm52(expand52)
    batch52 = BatchNormalization()(mid52)
    relu52 = Activation("relu")(batch52)
    drop5 = Dropout(0.5)(relu52)


    up1 = deconv1(drop5)
    merge6 = concatenate([drop4,up1], axis = 3)
    mid61 = conv61(merge6)
    batch61 = BatchNormalization()(mid61)
    relu61 = Activation("relu")(batch61)
    mid62 = conv62(relu61)
    batch62 = BatchNormalization()(mid62)
    relu62 = Activation("relu")(batch62)


    up2 = deconv2(relu62)
    merge7 = concatenate([relu32,up2], axis = 3)
    mid71 = conv71(merge7)
    batch71 = BatchNormalization()(mid71)
    relu71 = Activation("relu")(batch71)
    mid72 = conv72(relu71)
    batch72 = BatchNormalization()(mid72)
    relu72 = Activation("relu")(batch72)


    up3 = deconv3(relu72)
    merge8 = concatenate([relu22,up3], axis = 3)
    expand81 = Lambda(lambda x: keras.expand_dims(merge8, axis = 1))(merge8)
    mid81 = convlstm81(expand81)
    batch81 = BatchNormalization()(mid81)
    relu81 = Activation("relu")(batch81)
    expand82 = Lambda(lambda x: keras.expand_dims(relu81, axis = 1))(relu81)
    mid82 = convlstm82(expand82)
    batch82 = BatchNormalization()(mid82)
    relu82 = Activation("relu")(batch82)


    up4 = deconv4(relu82)
    merge9 = concatenate([relu12,up4], axis = 3)
    expand91 = Lambda(lambda x: keras.expand_dims(merge9, axis = 1))(merge9)
    mid91 = convlstm91(expand91)
    batch91 = BatchNormalization()(mid91)
    relu91 = Activation("relu")(batch91)
    expand92 = Lambda(lambda x: keras.expand_dims(relu91, axis = 1))(relu91)
    mid92 = convlstm92(expand92)
    batch92 = BatchNormalization()(mid92)
    relu92 = Activation("relu")(batch92)
    mid93 = conv93(relu92)
    batch93 = BatchNormalization()(mid93)
    relu93 = Activation("relu")(batch93)

    last = conv10(relu93)

    last3 = Add()([last, inputs])
    last1 = Subtract()([last3, inputs])

    #####second round#####

    expand11 = Lambda(lambda x: keras.expand_dims(last, axis = 1))(last)
    mid11 = convlstm11(expand11)
    batch11 = BatchNormalization()(mid11)
    relu11 = Activation("relu")(batch11)
    expand12 = Lambda(lambda x: keras.expand_dims(relu11, axis = 1))(relu11)
    mid12 = convlstm12(expand12)
    batch12 = BatchNormalization()(mid12)
    relu12 = Activation("relu")(batch12)
    pool1 = MaxPooling2D(pool_size=(2, 2))(relu12)


    expand21 = Lambda(lambda x: keras.expand_dims(pool1, axis = 1))(pool1)
    mid21 = convlstm21(expand21)
    batch21 = BatchNormalization()(mid21)
    relu21 = Activation("relu")(batch21)
    expand22 = Lambda(lambda x: keras.expand_dims(relu21, axis = 1))(relu21)
    mid22 = convlstm22(expand22)
    batch22 = BatchNormalization()(mid22)
    relu22 = Activation("relu")(batch22)
    pool2 = MaxPooling2D(pool_size=(2, 2))(relu22)


    mid31 = conv31(pool2)
    batch31 = BatchNormalization()(mid31)
    relu31 = Activation("relu")(batch31)
    mid32 = conv32(relu31)
    batch32 = BatchNormalization()(mid32)
    relu32 = Activation("relu")(batch32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(relu32)


    mid41 = conv41(pool3)
    batch41 = BatchNormalization()(mid41)
    relu41 = Activation("relu")(batch41)
    mid42 = conv42(relu41)
    batch42 = BatchNormalization()(mid42)
    relu42 = Activation("relu")(batch42)
    drop4 = Dropout(0.5)(relu42)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


    expand51 = Lambda(lambda x: keras.expand_dims(pool4, axis = 1))(pool4)
    mid51 = convlstm51(expand51)
    batch51 = BatchNormalization()(mid51)
    relu51 = Activation("relu")(batch51)
    expand52 = Lambda(lambda x: keras.expand_dims(relu51, axis = 1))(relu51)
    mid52 = convlstm52(expand52)
    batch52 = BatchNormalization()(mid52)
    relu52 = Activation("relu")(batch52)
    drop5 = Dropout(0.5)(relu52)


    up1 = deconv1(drop5)
    merge6 = concatenate([drop4,up1], axis = 3)
    mid61 = conv61(merge6)
    batch61 = BatchNormalization()(mid61)
    relu61 = Activation("relu")(batch61)
    mid62 = conv62(relu61)
    batch62 = BatchNormalization()(mid62)
    relu62 = Activation("relu")(batch62)


    up2 = deconv2(relu62)
    merge7 = concatenate([relu32,up2], axis = 3)
    mid71 = conv71(merge7)
    batch71 = BatchNormalization()(mid71)
    relu71 = Activation("relu")(batch71)
    mid72 = conv72(relu71)
    batch72 = BatchNormalization()(mid72)
    relu72 = Activation("relu")(batch72)


    up3 = deconv3(relu72)
    merge8 = concatenate([relu22,up3], axis = 3)
    expand81 = Lambda(lambda x: keras.expand_dims(merge8, axis = 1))(merge8)
    mid81 = convlstm81(expand81)
    batch81 = BatchNormalization()(mid81)
    relu81 = Activation("relu")(batch81)
    expand82 = Lambda(lambda x: keras.expand_dims(relu81, axis = 1))(relu81)
    mid82 = convlstm82(expand82)
    batch82 = BatchNormalization()(mid82)
    relu82 = Activation("relu")(batch82)


    up4 = deconv4(relu82)
    merge9 = concatenate([relu12,up4], axis = 3)
    expand91 = Lambda(lambda x: keras.expand_dims(merge9, axis = 1))(merge9)
    mid91 = convlstm91(expand91)
    batch91 = BatchNormalization()(mid91)
    relu91 = Activation("relu")(batch91)
    expand92 = Lambda(lambda x: keras.expand_dims(relu91, axis = 1))(relu91)
    mid92 = convlstm92(expand92)
    batch92 = BatchNormalization()(mid92)
    relu92 = Activation("relu")(batch92)
    mid93 = conv93(relu92)
    batch93 = BatchNormalization()(mid93)
    relu93 = Activation("relu")(batch93)

    last2 = conv10(relu93)

    model = Model(input = inputs, output = [last1,last2])
    model.compile(optimizer = Adam(lr = 1e-4),
                loss = 'categorical_crossentropy',
                loss_weights=[0.5, 1.0],
                metrics = [mean_iou])
    model.summary()

    return model