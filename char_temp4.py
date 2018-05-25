import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
K.set_image_dim_ordering('tf')
import time
import argparse
from scipy.io import loadmat
import pickle
#from PIL import Image
import random
import struct
np.random.seed(1337)
#data pre
'''
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
def dropconnect(W, p):
    return K.dropout(W, keep_prob=p) * p
x_train = read_idx('data/emnist-byclass-train-images-idx3-ubyte')
y_train = read_idx('data/emnist-byclass-train-labels-idx1-ubyte')
x_test= read_idx('data/emnist-byclass-test-images-idx3-ubyte')
y_test = read_idx('data/emnist-byclass-test-labels-idx1-ubyte')

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255
'''
def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.

        Arguments:
            mat_file_path: path to the .mat, should be in sample/

        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing

        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)

    '''
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)
(x_train,y_train),(x_test,y_test),mapping,number_of_classes=load_data('emnist-byclass.mat')
epochs=5
batch_size = 64
num_fil=128
mod_fil='check1.h5'
csv_1='check1.csv'
csv_2='check-pred.csv'
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

y_train[0], Y_train[0]
# Three steps to Convolution
# 1. Convolution
# 2. Activation
# 3. Polling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples
#data pre
model = Sequential()
model.add(Conv2D(num_fil,(5, 5),padding='valid', input_shape=(28,28,1),kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
BatchNormalization(axis=-1)
#model.add(Dropout(0.1))
model.add(Conv2D(num_fil, (5, 5),kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
model.add(AveragePooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
#model.add(Dropout(0.2))
#num_fil*=2
model.add(Conv2D(num_fil,(5, 5),padding='valid',kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
BatchNormalization(axis=-1)
#model.add(Dropout(0.3))
model.add(Conv2D(num_fil, (5, 5),kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
#model.add(Dropout(0.4))
model.add(Flatten())
# Fully connected layer

model.add(Dense(1024,kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
#BatchNormalization(axis=-1)
model.add(Dropout(0.5))
model.add(Dense(512,kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
#BatchNormalization(axis=-1)
model.add(Dropout(0.5))
model.add(Dense(256, kernel_initializer='he_normal'))
model.add(PReLU(weights=None, alpha_initializer="zero"))
#BatchNormalization(axis=-1)
model.add(Dropout(0.5))

model.add(Dense(number_of_classes))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.9, nesterov=True), metrics=['accuracy'])

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()
train_generator = gen.flow(x_train, Y_train, batch_size=batch_size)
test_generator = test_gen.flow(x_test, Y_test, batch_size=batch_size)
# model.fit(X_train, Y_train, batch_size=128, nb_epoch=1, validation_data=(X_test, Y_test))
t0=time.time()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
model.fit_generator(train_generator, steps_per_epoch=len(x_train)//batch_size, epochs=epochs, 
                    validation_data=test_generator, validation_steps=len(x_test)//batch_size,callbacks=[reduce_lr])
model.save(mod_fil)
t1=time.time()
print("Training completed in " + str(t1-t0) + " seconds")
model_yaml = model.to_yaml()
with open("bin/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

score = model.evaluate(x_test, Y_test)
print('Test loss:', score[0])
print('Test accuracy: ', score[1])
predictions = model.predict_classes(x_test)

predictions = list(predictions)
actuals = list(y_test)
p=model.predict(x_test)
sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
#acc,loss,val_acc,val_loss=model.history.history
#sub.to_csv(csv_1, index=False)
np.savetxt(csv_1, np.c_[range(1,len(p)+1),p], delimiter=',', comments = '', fmt='%f')
np.savetxt(csv_2, np.c_[range(1,len(predictions)+1),predictions,actuals], delimiter=',', header = 'ImageId,Label,Actual', comments = '', fmt='%d')

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
model = load_model(mod_fil)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(momentum=0.9, nesterov=True),
              metrics=['accuracy'])
model_yaml = model.to_yaml()
with open("bin/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
import matplotlib.pyplot as plt
plt.plot(model.history)
img = image.load_img('hsf_0_00005.png', target_size=(28, 28),grayscale=True)
img = x_test.reshape(img.shape, 28, 28, 1)
with open("hsf_0_00005.png",'rb') as imageFile:
  f = imageFile.read()
  b = bytearray(f)
img/=255
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=batch_size)
print (classes)

'''
num_classes=10
img_size=28
fil_num=128
batch_size=480
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(fil_num,(3, 3), input_shape = (img_size, img_size, 1)))
classifier.add(PReLU(weights=None, alpha_initializer="zero"))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(fil_num, (3, 3)))
classifier.add(PReLU(weights=None, alpha_initializer="zero"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(fil_num, (3, 3)))
classifier.add(PReLU(weights=None, alpha_initializer="zero"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128))
classifier.add(PReLU(weights=None, alpha_initializer="zero"))
classifier.add(Dense(activation="softmax", units=10))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


datagen.fit(x_train)
training_set = datagen.flow(x_train, y_train,batch_size=batch_size)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_size, img_size),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_size, img_size),
                                            batch_size = batch_size,
                                            class_mode = 'binary')


datagen.fit(x_test)
test_set = datagen.flow(x_test, y_test,batch_size=batch_size)

t0=time.time()
classifier.fit_generator(training_set,
                         steps_per_epoch = 125,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 21)
t1=time.time()
print("Training completed in " + str(t1-t0) + " seconds")
classifier.save('digit_1.h5')
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images

# load the model we saved
model = load_model('model_7.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''