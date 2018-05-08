# import libraries
import json
from pprint import pprint
from pandas.io.json import json_normalize
import pandas as pd
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os, sys, glob
#import matplotlib
import math
import random
from keras.optimizers import Optimizer
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.legacy import interfaces
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
import re

# explore the person keypoints json file associated with the coco 2017 dataset
keypoints_train = json.load(open('./dataset/annotations/person_keypoints_train2017.json'))
keypoints_val = json.load(open('./dataset/annotations/person_keypoints_val2017.json'))


df_train_category = json_normalize(keypoints_train['categories'])
skeleton = df_train_category['skeleton'][0]
KPs = df_train_category['keypoints'][0]

df_train_image = json_normalize(keypoints_train['images'])
df_train_annotation = json_normalize(keypoints_train['annotations'])
df_val_image = json_normalize(keypoints_val['images'])
df_val_annotation = json_normalize(keypoints_val['annotations'])


# print the number of labeled object in the train and validation dataset
print ("the number of data objects in the trainging set is %6.2d." % df_train_annotation.shape[0])
print ("the number of data objects in the validation set is %6.2d." % df_val_annotation.shape[0])
# print the number of images in the train and validation dataset
print ("\nthe number of images in the trainging set is %6.2d." % df_train_annotation['image_id'].unique().shape[0])
print ("the number of images in the validation set is %6.2d." % df_val_annotation['image_id'].unique().shape[0])

# merge the image and annotation table on image_id
df_train_image = df_train_image.rename(index=str, columns={"id": "image_id"})
df_train_image = pd.merge(df_train_image, df_train_annotation, on='image_id', how='left')
df_val_image = df_val_image.rename(index=str, columns={"id": "image_id"})
df_val_image = pd.merge(df_val_image, df_val_annotation, on='image_id', how='left')


# extract the relevant information based on the number of visible human key-points
keypoints_thredhold = 15
df_train_raw = df_train_image[(df_train_image['num_keypoints']>keypoints_thredhold)][['file_name','height','image_id','width','bbox','keypoints',\
                                                                      'num_keypoints','segmentation']].reset_index(drop=True)
df_val = df_val_image[df_val_image['num_keypoints']>keypoints_thredhold][['file_name','height','image_id','width','bbox','keypoints',\
                                                                      'num_keypoints','segmentation']].reset_index(drop=True)


# print the number of labeled object in the train and validation dataset
print ('After data filtration by removing the non-human objects and images\n')
print ("the number of data objects in the trainging set is %6.2d." % df_train_raw.shape[0])
print ("the number of data objects in the validation set is %6.2d." % df_val.shape[0])
# print the number of images in the train and validation dataset
print ("\nthe number of images in the trainging set is %6.2d." % df_train_raw['image_id'].unique().shape[0])
print ("the number of images in the validation set is %6.2d." % df_val['image_id'].unique().shape[0])


df_train = df_train_raw.sample(frac=1.00).reset_index()
df_test = df_train_raw.sample(frac=0.1).reset_index()


# print the number of labeled object in the train and validation dataset
print ('Subsampling\n')
print ("the number of data objects in the trainging set is %6.2d." % df_train.shape[0])
print ("the number of data objects in the validation set is %6.2d." % df_val.shape[0])
print ("the number of data objects in the test set is %6.2d." % df_test.shape[0])
# print the number of images in the train and validation dataset
print ("\nthe number of images in the trainging set is %6.2d." % df_train['image_id'].unique().shape[0])
print ("the number of images in the validation set is %6.2d." % df_val['image_id'].unique().shape[0])
print ("the number of images in the test set is %6.2d." % df_test['image_id'].unique().shape[0])



# find the text after a first string
def find_after( s, first):
    try:
        start = s.index( first ) + len( first )
        #end = s.index( last, start )
        return s[start:]
    except ValueError:
        return ""


Raw_images = []
file_path = []
for i in df_train['file_name']:
    file = './dataset/train2017/' + i
    file_path.append(file)


# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
#cmap = matplotlib.cm.get_cmap('hsv')

def Gaussian2d(x, y, x0, y0, a, sigma=10):
    xx = (float(x) - x0)** 2 / 2 / sigma **2
    yy = (float(y) - y0)** 2 / 2 / sigma **2
    return a * math.exp(- xx - yy)


def generator(batch_size):
    n_record = len(df_train)
    # Create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, 220, 300, 3))
    batch_label = np.zeros((batch_size, 27, 37, 17))
    while True:
        batch_labels = []
        for i in range(batch_size):
            # choose random index in features
            index= random.randint(0,n_record)
            img = cv2.imread(file_path[index])
            # create one layer for each of labeled part
            w,h = img.shape[0], img.shape[1]
            heat_map = create_labels(img, df_train['keypoints'][index])
            process_img = img_processing(img)
            batch_features[i] = process_img
            batch_label[i] = heat_map
        print ("batch feed step done")
        print (batch_features[0].shape)
        print (batch_label[0].shape)
        for i in range(6):
            batch_labels.append(batch_label)
        yield batch_features, batch_labels

        
def img_processing(img):
    # data processing and data augumentation
    # TODO, data augmentation
    img = img/255. - 0.5
    img = cv2.resize(img,(300, 220),interpolation=cv2.INTER_CUBIC)
    return img
    

def create_labels(img, points):
    # assume the raw images will be resized in to (220, 300)
    # the heat_map will be in the size of (27,37)
    w, h = img.shape[0], img.shape[1]
    heat_map_raw = np.zeros((w, h, 17))
    radius = 50
    for i in range (17):
        if points[i*3+2] == 2:
            x0,y0 = points[i*3+1], points[i*3]
            for x in range (x0-radius, x0+radius):
                for y in range (y0-radius, y0+radius ):
                    if x in range (0,w) and y in range (0,h):
                        heat_map_raw[x, y, i] = Gaussian2d(x,y, x0,y0, 1, 10)
    heat_map = cv2.resize(heat_map_raw, (37,27),interpolation=cv2.INTER_CUBIC)

    return heat_map


def relu(x): 
    return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)

    return x


def stage1_block(x, num_p, branch, weight_decay):

    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    return x


def stageT_block(x, num_p, stage, branch, weight_decay):

    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    return x


def get_training_model(weight_decay, np_branch2, stages = 6, gpus = None):

    img_input_shape = (None, None, 3)
    heat_input_shape = (None, None, np_branch2)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)

    #inputs.append(heat_weight_input)

    #img_normalized = img_input # will be done on augmentation stage

    # VGG
    stage0_out = vgg_block(img_input, weight_decay)

    # stage 1
    new_x = []
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w = Multiply()([stage1_branch2_out, heat_weight_input])
    #w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2, np_branch1, np_branch2)
    outputs.append(stage1_branch2_out)
    new_x.append(stage1_branch2_out)

    new_x.append(stage0_out)

    x = Concatenate()(new_x)

    # stage sn >= 2, repeating stage
    for sn in range(2, stages + 1):
         new_x = []
         # stage T
         stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
         w = Multiply()([stageT_branch2_out, heat_weight_input])
         #w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2, np_branch1, np_branch2)
         outputs.append(stageT_branch2_out)
         new_x.append(stageT_branch2_out)
         new_x.append(stage0_out)

         if sn < stages:
             x = Concatenate()(new_x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_lrmult(model):

    # setup lr multipliers for conv layers
    lr_mult = dict()

    for layer in model.layers:

        if isinstance(layer, Conv2D):

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
               print("matched as vgg layer", layer.name)
               kernel_name = layer.weights[0].name
               bias_name = layer.weights[1].name
               lr_mult[kernel_name] = 1
               lr_mult[bias_name] = 2

    return lr_mult


batch_size = 16
base_lr = 2e-4
momentum = 0.9
weight_decay = 5e-4
lr_policy = "step"
gamma = 0.333
stepsize = 10000 * 17  # in original code each epoch is 121746 and step change is on 17th epoch
max_iter = 200

file_name = 'V1'
date = '05082018'
file_id = file_name + "_" + date

WEIGHT_DIR = "./" + file_id
WEIGHTS_SAVE = 'weights_V1.h5'

TRAINING_LOG = "./" + file_id + ".csv"
LOGS_DIR = "./logs"

model = get_training_model(weight_decay,np_branch2=17)
lr_mult = get_lrmult(model)

# Initialized the first 10 vgg layers using the pre-trained weights from vgg19

print("Loading vgg19 weights...")

vgg_model = VGG19(include_top=False, weights='imagenet')

from_vgg = dict()
from_vgg['conv1_1'] = 'block1_conv1'
from_vgg['conv1_2'] = 'block1_conv2'
from_vgg['conv2_1'] = 'block2_conv1'
from_vgg['conv2_2'] = 'block2_conv2'
from_vgg['conv3_1'] = 'block3_conv1'
from_vgg['conv3_2'] = 'block3_conv2'
from_vgg['conv3_3'] = 'block3_conv3'
from_vgg['conv3_4'] = 'block3_conv4'
from_vgg['conv4_1'] = 'block4_conv1'
from_vgg['conv4_2'] = 'block4_conv2'


for layer in model.layers:
    if layer.name in from_vgg:
        vgg_layer_name = from_vgg[layer.name]
        layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
        print("Loaded VGG19 layer: " + vgg_layer_name)

last_epoch = 0

# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    l = K.sum(K.square(x - y)) / batch_size / 2
    return l

# learning rate schedule - equivalent of caffe lr_policy =  "step"
iterations_per_epoch = len(df_train) // batch_size

def step_decay(epoch):
    steps = epoch * iterations_per_epoch * batch_size
    lrate = base_lr * math.pow(gamma, math.floor(steps/stepsize))
    print("Epoch:", epoch, "Learning rate:", lrate)
    return lrate

print("Weight decay policy...")
for i in range(1,100,5): step_decay(i)



# configure callbacks
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(WEIGHT_DIR + '/' + WEIGHTS_SAVE, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)
tnan = TerminateOnNaN()
#coco_eval = CocoEval(train_client, val_client)

callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan]

csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])

class MultiSGD(Optimizer):
    """
    Modified SGD with added support for learning multiplier for kernels and biases
    as suggested in: https://github.com/fchollet/keras/issues/5920
    Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, lr_mult=None, **kwargs):
        super(MultiSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_mult = lr_mult

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):

            if p.name in self.lr_mult:
                multiplied_lr = lr * self.lr_mult[p.name]
            else:
                multiplied_lr = lr

            v = self.momentum * m - multiplied_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - multiplied_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(MultiSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)
# start training
model.compile(loss=eucl_loss, optimizer=multisgd)
model.save('my_model_V1.h5')  # creates a HDF5 file 'my_model.h5'

model.fit_generator(generator(batch_size), steps_per_epoch = 100, epochs=max_iter, callbacks=callbacks_list)

model.load_weights('model_weights_V1.h5')
print ('model saving done')






