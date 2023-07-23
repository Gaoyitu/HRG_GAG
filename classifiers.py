import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.models import Sequential



def get_mnist_classifier_c1():

    inputs = Input(shape=(28, 28, 1))
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation="relu", name="conv_1")(inputs)
    max_pooling_1 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_1)
    conv_2 = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv_2")(max_pooling_1)
    max_pooling_2 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_2)

    max_pooling_2_flat = Flatten(name='flatten')(max_pooling_2)

    fc_1 = Dense(200, activation="relu",name='feature_layer')(max_pooling_2_flat)

    outputs = Dense(10, activation='softmax')(fc_1)

    model = Model(inputs=inputs, outputs=outputs)

    model.load_weights('networks/mnist_c1.h5')


    return model


def get_mnist_classifier_c2():

    inputs = Input(shape=(28, 28, 1))
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation="relu", name="conv_1")(inputs)
    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation="relu", name="conv_2")(conv_1)
    max_pooling_1 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_2)
    conv_3 = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv_3")(max_pooling_1)
    conv_4 = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv_4")(conv_3)
    max_pooling_2 = MaxPooling2D((2, 2), (2, 2), padding="same")(conv_4)

    max_pooling_2_flat = Flatten(name='flatten')(max_pooling_2)

    fc_1 = Dense(200, activation="relu",name='fc_1')(max_pooling_2_flat)
    fc_2 = Dense(200, activation="relu", name='fc_2')(fc_1)

    outputs = Dense(10, activation='softmax')(fc_2)


    model = Model(inputs=inputs, outputs=outputs)

    model.load_weights('networks/mnist_c2.h5')

    return model


def get_cifar10_vgg16():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    model.add(Dense(10, activation="softmax"))

    model.load_weights('networks/cifar10_vgg16.h5')

    return model



def get_cifar10_classifier_c3():
    inputs = Input(shape=(32, 32, 3))
    conv_1 = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    activation="relu",
                    name="conv_1",
                    kernel_initializer='glorot_uniform')(inputs)
    conv_2 = Conv2D(64, (3, 3),
                    padding='same',
                    activation="relu",
                    name="conv_2",
                    kernel_initializer='glorot_uniform')(conv_1)
    max_pooling_1 = MaxPooling2D((2, 2), (2, 2),
                                  padding="same",name="pool1")(conv_2)
    conv_3 = Conv2D(128, (3, 3),
                    padding='same',
                    activation="relu",
                    name="conv_3",
                    kernel_initializer='glorot_uniform')(max_pooling_1)
    conv_4 = Conv2D(128, (3, 3),
                    padding='same',
                    activation="relu",
                    name="conv_4",
                    kernel_initializer='glorot_uniform')(conv_3)
    max_pooling_2 = MaxPooling2D((2, 2), (2, 2),
                                  padding="same",name="pool2")(conv_4)

    max_pooling_2_flat = Flatten(name='flatten')(max_pooling_2)

    fc_1 = Dense(256,
                 activation="relu",
                 kernel_initializer='he_normal',name='fc_1')(max_pooling_2_flat)

    fc_2 = Dense(256,
                 activation="relu",
                 kernel_initializer='he_normal',name = 'fc_2')(fc_1)

    outputs = Dense(10, activation='softmax')(fc_2)

    model = Model(inputs=inputs, outputs=outputs)

    model.load_weights('networks/cifar10_c3.h5')

    return model


def get_imagenet_vgg16():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    model.add(Dense(10, activation="softmax"))

    model.load_weights('networks/imagenet_vgg16.h5')

    return model



def get_imagenet_mobilenet():
    conv_base = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    model.add(Dense(10, activation='softmax'))
    conv_base.trainable = False

    model.load_weights('networks/imagenet_mobilenet.h5')

    return model


def get_imagenet_64_vgg16():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(64,64, 3))

    model = Sequential()

    model.add(conv_base)
    model.add(Flatten(name='flatten'))

    # model.add(Dense(10, activation="softmax"))
    model.add(Dense(10, activation=None))

    model.load_weights('./networks/imagenet/imagenet_64_vgg16_weights.h5')

    return model













