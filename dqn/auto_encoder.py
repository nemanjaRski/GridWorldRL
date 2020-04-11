from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg
import cv2


def load_data(path):
    data = []
    for filename in os.listdir(f'{path}'):
        img_name = f'{path}/{filename}'
        img = mpimg.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = np.expand_dims(img, axis=0)
        data.append(img)
    X_train, X_test = np.array(data[:8000]), np.array(data[8000])
    return X_train, X_test

input_img = Input(shape=[84, 84, 3], name="input")  # adapt this if using `channels_first` image data format

""" Original architecture
x = Conv2D(32, kernel_size=[8, 8], strides=[4, 4], activation='relu', padding='valid', name="conv_1")(input_img)
x = Conv2D(64, kernel_size=[4, 4], strides=[2, 2], activation='relu', padding='valid', name="conv_2")(x)
x = Conv2D(64, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='valid', name="conv_3")(x)
encoded = Conv2D(512, kernel_size=[7, 7], strides=[1, 1], activation='relu', padding='valid', name="conv_4")(x)

x = Conv2D(512, kernel_size=[7, 7], strides=[1, 1], activation='relu', padding='valid', name="conv_4")(encoded)
x = Conv2D(64, kernel_size=[3, 3], strides=[1, 1], activation='relu', padding='valid', name="conv_3")(x)
x = Conv2D(64, kernel_size=[4, 4], strides=[2, 2], activation='relu', padding='valid', name="conv_2")(x)
x = Conv2D(32, kernel_size=[8, 8], strides=[4, 4], activation='relu', padding='valid', name="conv_1")(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
"""

""" First test example
x = Conv2D(32, kernel_size=[5, 5], activation='relu', padding='valid', name="conv_1")(input_img)
x = MaxPooling2D((2, 2), padding='valid')(x)
x = Conv2D(64, kernel_size=[3, 3], activation='relu', padding='valid', name="conv_2")(x)
x = MaxPooling2D((2, 2), padding='valid')(x)
x = Conv2D(64, kernel_size=[3, 3], activation='relu', padding='valid', name="conv_3")(x)
x = MaxPooling2D((2, 2), padding='valid')(x)
x = Conv2D(64, kernel_size=[5, 5], activation='relu', padding='valid', name="conv_4")(x)
encoded = MaxPooling2D((2, 2), padding='valid')(x)

x = Conv2D(64, kernel_size=[5, 5], activation='relu', padding='valid', name="deconv_1")(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, kernel_size=[3, 3], activation='relu', padding='valid', name="deconv_2")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, kernel_size=[3, 3], activation='relu', padding='valid', name="deconv_3")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, kernel_size=[5, 5], activation='relu', padding='valid', name="deconv_4")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
"""

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

x_train, x_test = load_data("conv_images")
x_train = np.reshape(x_train, (len(x_train), 84, 84, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 84, 84, 3))  # adapt this if using `channels_first` image data format


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save_weights("auto_encoder.h5")
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(84, 84))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(84, 84))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
