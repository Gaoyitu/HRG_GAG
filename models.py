
from tensorflow import  add
import numpy as np
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,concatenate,BatchNormalization,ReLU,LeakyReLU,UpSampling2D,\
    MaxPool2D,Flatten,Dense,Reshape,Dropout,Conv2DTranspose,GaussianNoise


def get_mnist_generator_HRG():

    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (3, 3), (1, 1), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)

    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)

    x = add(x, np.random.normal(0, 0.1, (64)))

    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)

    x = Reshape((2, 2, 512))(x)

    x = Conv2DTranspose(256, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(32, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(1, (5, 5), (1, 1), padding='valid',
               activation='tanh', kernel_initializer='glorot_uniform')(x)

    autoencoder = Model(inputs=inputs, outputs=x)
    return autoencoder



def get_mnist_generator_GAG():
    inputs = Input(shape=(28,28,1))

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = GaussianNoise(0.1)(x)


    x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(1, 1), padding='valid',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2DTranspose(128, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2DTranspose(64, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2D(1, (5, 5), (1, 1), padding='same',
               activation='tanh', kernel_initializer='glorot_uniform')(x)

    autoencoder = Model(inputs=inputs, outputs=x)
    return autoencoder

def get_mnist_discriminator():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(inputs)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    S_out = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', use_bias=False)(x)
    C_out = Dense(10, activation='softmax', kernel_initializer='glorot_uniform', use_bias=False)(x)

    model = Model(inputs=inputs, outputs=[S_out, C_out])

    return model


def get_cifar10_generator_HRG():
    inputs = Input(shape=(32, 32, 13))

    x = Conv2D(128, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1024, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(1024, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(512, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(256, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, (3, 3), (2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(1, (5, 5), (1, 1), padding='same',
               activation='tanh', kernel_initializer='glorot_uniform')(x)

    autoencoder = Model(inputs=inputs, outputs=x)
    return autoencoder

def get_cifar10_generator_GAG():
    inputs = Input(shape=(32, 32, 3))

    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = GaussianNoise(0.1)(x)


    x = Conv2DTranspose(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                        activation=None, kernel_initializer='glorot_uniform', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = Conv2D(3, (5, 5), (1, 1), padding='same',
               activation='tanh', kernel_initializer='glorot_uniform')(x)

    autoencoder = Model(inputs=inputs, outputs=x)
    return autoencoder

def get_cifar10_discriminator():
    inputs = Input(shape=(32, 32, 3))

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=True)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Conv2D(512, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform', use_bias=True)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    S_out = Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform', use_bias=False)(x)
    C_out = Dense(10, activation='softmax', kernel_initializer='glorot_uniform', use_bias=False)(x)

    model = Model(inputs=inputs, outputs=[S_out, C_out])

    return model



def get_imagenet_generator_HRG():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1024, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1024, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = add(x, np.random.normal(0, 0.1, (7,7,1024)))

    x = Conv2DTranspose(1024, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(1024, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(512, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(256, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, (3, 3), (2, 2), padding='same',
               activation=None, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(3, (5, 5), (1, 1), padding='same',
               activation='tanh', kernel_initializer='glorot_uniform')(x)

    autoencoder = Model(inputs=inputs, outputs=x)

    return autoencoder



if __name__ == '__main__':
    get_mnist_generator_HRG().summary()