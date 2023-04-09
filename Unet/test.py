import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, Reshape, concatenate
from keras.models import Model
from keras.layers import BatchNormalization, Dropout, Activation, MaxPool2D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from IPython.display import Image
from skimage import color

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')
# %matplotlib inline

SEED = 34

#colab연동
# from google.colab import drive
# drive.mount('/gdrive', force_remount=True)
# !ls -al /gdrive/'My Drive'/'Colab Notebooks'/data/pfcn_small.npz

pfcn_small = np.load('pfcn_small.npz')
train_images = pfcn_small['train_images']
train_mattes = pfcn_small['train_mattes']
test_images = pfcn_small['test_images']
test_mattes = pfcn_small['test_mattes']

print(train_images.shape)
print(train_mattes.shape)
print(test_images.shape)
print(test_mattes.shape)
train_images.dtype
# plt.imshow(train_images[0])
# plt.show()
# plt.imshow(train_mattes[0])
# plt.show()
print(train_images.max(), train_images.min())
print(test_images.max(), test_images.min())
# Convert grayscale images to RGB by duplicating the single channel
train_mattes = np.array([color.gray2rgb(img) for img in train_mattes])
test_mattes = np.array([color.gray2rgb(img) for img in test_mattes])

# Convert RGB images to grayscale and add an additional channel
train_mattes = np.array([color.rgb2gray(img).reshape((100, 75, 1)) for img in train_mattes])
test_mattes = np.array([color.rgb2gray(img).reshape((100, 75, 1)) for img in test_mattes])

# Check the shape of the converted data
train_mattes.shape, test_mattes.shape
plt.imshow(train_images[:5].transpose([1, 0, 2, 3]).reshape((100,-1,3)))
plt.show()
plt.imshow(train_mattes[:5].transpose([1, 0, 2, 3]).reshape((100,-1)), cmap = 'gray')
plt.show()

#간단한 U-net모델 생성

def conv2d_block(x, channel):
  x = Conv2D(channel, 3, padding = 'same')(x)
                      #커널
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(channel, 3, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x

def unet_like():
  inputs = Input((100, 75, 3))

  c1 = conv2d_block(inputs, 16)
  p1 = MaxPool2D((2,2))(c1)
  p1 = Dropout(0.1)(p1)

  c2 = conv2d_block(p1, 32)
  p2 = MaxPool2D((2,2))(c2)
  p2 = Dropout(0.1)(p2)

  c3 = conv2d_block(p2, 64)
  p3 = MaxPool2D((2,2))(c3)
  p3 = Dropout(0.1)(p3)

  c4 = conv2d_block(p3, 128)
  p4 = MaxPool2D((2,2))(c4)
  p4 = Dropout(0.1)(p4)

  c5 = conv2d_block(p4, 256)

  u6 = Conv2DTranspose(128, 2, 2, output_padding=(0,1))(c5)
  u6 = concatenate([u6, c4]) # 사이즈 128
  u6 = Dropout(0.1)(u6)
  c6 = conv2d_block(u6, 128)

  u7 = Conv2DTranspose(64, 2, 2, output_padding=(1,0))(c6)
  u7 = concatenate([u7, c3]) 
  u7 = Dropout(0.1)(u7)
  c7 = conv2d_block(u7, 64)

  u8 = Conv2DTranspose(32, 2, 2, output_padding=(0,1))(c7)
  u8 = concatenate([u8, c2])
  u8 = Dropout(0.1)(u8)
  c8 = conv2d_block(u8, 32)

  u9 = Conv2DTranspose(16, 2, 2, output_padding=(0,1))(c8)
  u9 = concatenate([u9, c1]) 
  u9 = Dropout(0.1)(u9)
  c9 = conv2d_block(u9, 16)

  outputs = Conv2D(1, (1,1), activation = 'sigmoid')(c9)

  model = Model(inputs, outputs)
  return model
  
model = unet_like()
model.summary()
model.compile(loss = 'mse', optimizer='adam', metrics = ['accuracy'])
hist = model.fit(train_images, train_mattes, validation_data=(test_images, test_mattes), epochs = 25, verbose = 1)

plt.plot(hist.history['accuracy'], label = 'accuracy')
plt.plot(hist.history['loss'], label = 'loss')
plt.plot(hist.history['val_accuracy'], label = 'val_accuracy')
plt.plot(hist.history['val_loss'], label = 'val_loss')
plt.legend(loc = 'right')
plt.show()
res = model.predict(test_images[0:1])
plt.imshow(np.concatenate([res[0], test_mattes[0]]).reshape((2,-1,75,1)).transpose([1,0,2,3]).reshape((100,-1)), cmap = 'gray')
plt.show()
imgs = np.concatenate([(res>0.5).astype(np.float64).reshape((100,75,1)), test_mattes[0]]).reshape((2,-1,75,1)).transpose((1,0,2,3)).reshape((100,-1))
plt.imshow(imgs, cmap = 'gray')
plt.show() 
plt.figure(figsize = (8,8))
plt.subplot(121)
plt.imshow(test_images[0] * test_mattes[0].reshape((100,75,1)))

plt.subplot(122)
plt.imshow(test_images[0] * model.predict(test_images[0:1]).reshape((100,75,1)))

plt.show()

plt.figure(figsize = (8,8))
plt.subplot(121)
plt.imshow(test_images[1] * test_mattes[1].reshape((100,75,1)))

plt.subplot(122)
plt.imshow(test_images[1] * model.predict(test_images[1:2]).reshape((100,75,1)))

plt.show()

directory = "result"


if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(len(test_images)):
    result_image = test_images[i] * model.predict(test_images[i:i+1]).reshape((100,75,1))
    result_image = (result_image - result_image.min()) / (result_image.max() - result_image.min())
    result_image = (result_image * 255).astype(np.uint8)
    result_image = Image.fromarray(result_image)
    result_image.save(os.path.join(directory, f"result_{i}.png"))