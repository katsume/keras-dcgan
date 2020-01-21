from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow
import math
import os
import sys
import argparse
import uuid
import numpy as np

from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

g_opt = Adam(lr=2e-4, beta_1=0.5)
d_opt = Adam(lr=1e-5, beta_1=0.1)
z_dim= 100
num_images= 16
num_epoch= 20
batch_size= 32

def load_data(path, max):

    # file
    #
    filenames= os.listdir(path)
    x_train= []
    for i in range(min(max, len(filenames))):
        img= img_to_array(Image.open(path+filenames[i]))
        img= (img.astype(np.float32) - 127.5)/127.5
        x_train.append(img)
    x_train= np.array(x_train)

    # mnist from keras datasets
    #
    # from tensorflow.keras.datasets import mnist
    # (x_train, y_train), (_, _) = mnist.load_data()
    # x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    # cifar10 from keras datasets
    #
    # from tensorflow.keras.datasets import cifar10
    # (x_train, y_train), (_, _) = cifar10.load_data()
    # x_train = (x_train.astype(np.float32) - 127.5)/127.5

    return x_train

def generator_model(image_shape):
    data_format = 'channels_last'
    width= int(image_shape[0]/4)
    height= int(image_shape[1]/4)
    model = Sequential()

    model.add(Dense(
        input_shape=(z_dim,),
        units=1024,
        kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128*width*height))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Reshape((width, height, 128), input_shape=(128*width*height,)))
    model.add(UpSampling2D((2, 2), data_format=data_format))
    model.add(Conv2D(
        64, (5, 5),
        padding='same',
        data_format=data_format,
        kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(UpSampling2D((2, 2), data_format=data_format))
    model.add(Conv2D(
        image_shape[2], (5, 5),
        padding='same',
        data_format=data_format,
        kernel_initializer='he_normal'))
    model.add(Activation('tanh'))

    return model

def discriminator_model(image_shape):
    data_format = 'channels_last'
    model = Sequential()

    model.add(Conv2D(
        64, (5, 5),
        strides=(2, 2),
        padding='same',
        data_format=data_format,
        input_shape=image_shape,
        kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(
        128, (5, 5),
        strides=(2, 2),
        data_format=data_format,
        kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))

    model.add(Flatten())

    model.add(Dense(256,
        kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    width, height, ch= generated_images.shape[1:]
    output_shape = (height*rows, width*cols, ch)
    if ch<=1:
        output_shape = (height*rows, width*cols)
    output_image = np.zeros(output_shape, dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        if ch<=1:
            image= image[:, :, 0]
        output_image[width*i:width*(i+1), height*j:height*(j+1)] = image

    output_image= output_image*127.5 + 127.5

    return Image.fromarray(output_image.astype(np.uint8))

def train(batch_size, num_epoch, src_path, src_max, dst_path, num_images):

    x_train= load_data(src_path, src_max)
    src_shape= x_train[0].shape

    noise_fix= np.array([np.random.uniform(-1, 1, z_dim) for _ in range(int(num_images/2))])

    # generator
    generator = generator_model(src_shape)

    # discriminator
    discriminator = discriminator_model(src_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator+discriminator（discriminator部分の重みは固定）
    discriminator.trainable = False
    dcgan = Sequential([generator, discriminator])
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(x_train.shape[0] / batch_size)
    print('Number of batches:', num_batches)
    for epoch in range(num_epoch):

        for index in range(num_batches):

            noise = np.array([np.random.uniform(-1, 1, z_dim) for _ in range(batch_size)])
            image_batch = x_train[index*batch_size:(index+1)*batch_size]
            generated_images = generator.predict(noise, verbose=0)

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1]*batch_size + [0]*batch_size
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, z_dim) for _ in range(batch_size)])
            g_loss = dcgan.train_on_batch(noise, [1]*batch_size)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

        # 生成画像を出力
        fix_images = generator.predict(noise_fix, verbose=0)
        noise = np.array([np.random.uniform(-1, 1, z_dim) for _ in range(int(num_images/2))])
        var_images = generator.predict(noise, verbose=0)

        combined_image= combine_images(np.concatenate([fix_images, var_images]))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        combined_image.save(dst_path+"%04d.png" % (epoch))

        generator.save_weights('generator.h5')
        discriminator.save_weights('discriminator.h5')

def generate(batch_size, nice, dst_path):
    return

def get_args():
    parser= argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str)
    parser.add_argument('--epoch', type=int, default=num_epoch)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    # parser.add_argument('--nice', action='store_true')
    parser.add_argument('--src-path', type=str, default='')
    parser.add_argument('--src-max', type=int, default=sys.maxsize)
    parser.add_argument('--dst-path', type=str, default='dst/'+str(uuid.uuid4())+'/')
    parser.add_argument('--images', type=int, default=num_images)
    args= parser.parse_args()
    return args

if __name__=='__main__':
    args= get_args()

    train(
        batch_size= args.batch_size,
        num_epoch= args.epoch,
        src_path= args.src_path,
        src_max= args.src_max,
        dst_path= args.dst_path,
        num_images= args.images
    )

    # if args.mode=='train':
    #     train(
    #         batch_size= args.batch_size,
    #         num_epoch= args.epoch,
    #         src_path= args.src_path,
    #         src_max= args.src_max,
    #         dst_path= args.dst_path
    #     )
    # elif args.mode=='generate':
    #     generate(
    #         batch_size=args.batch_size,
    #         nice=args.nice,
    #         dst_path=args.dst_path
    #     )
