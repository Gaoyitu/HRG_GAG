
import tensorflow as tf
from tensorflow.keras.models import Sequential
import models
import classifiers
import data
import numpy as np

def training(x_train,y_train,x_test,y_test,batch_size,generator,classifier,save_path,loss):
    '''
    :param x_train: x in training set
    :param y_train: y in training set
    :param x_test: x in testing set
    :param y_test: y in testing set
    :param batch_size: batch size 512 for MNIST and CIFAR-10, 32 for ImageNet
    :param generator: generator
    :param classifier: classifier
    :param save_path: the patch of save weights of generators
    :param loss: Log-Likelihood, L2 Norm and KLD
    :return:
    '''
    train_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, verbose=1
        )
    ]

    model = Sequential()
    model.add(generator)
    model.add(classifier)
    classifier.trainable = False

    model.summary()

    if loss == 'crossentropy':
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=100, callbacks=train_callbacks,
                  validation_data=(x_test, y_test))

        generator.save_weights(save_path)

    if loss == 'mse':
        model.compile(optimizer='adam',
                      loss='mse')

        model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=100, callbacks=train_callbacks,
                  validation_data=(x_test, y_test))

        generator.save_weights(save_path)


    if loss == 'kld':
        model.compile(optimizer='adam',
                     loss='KLD')

        model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=100, callbacks=train_callbacks,
                  validation_data=(x_test, y_test))

        generator.save_weights(save_path)


def get_data(dataset,classifier_name, loss):
    '''
    :param dataset:
    :param classifier_name:
    :param loss:
    :return: training data and testing data
    '''
    if dataset == 'mnist':
        if loss == 'crossentropy':
            x_train, y_train, x_test, y_test = data.get_mnist()
        else:
            x_train, _, x_test, _ = data.get_mnist()
            y_train = np.load('data/'+dataset+'_' + classifier_name + '.npy')
            y_test = np.load('data/'+dataset+'_' + classifier_name + '_test.npy')
    if dataset == 'cifar10':
        if loss == 'crossentropy':
            x_train, y_train, x_test, y_test = data.get_cifar10()
        else:
            x_train, _, x_test, _ = data.get_cifar10()
            y_train = np.load('data/' + dataset + '_' + classifier_name + '.npy')
            y_test = np.load('data/' + dataset + '_' + classifier_name + '_test.npy')
    if dataset == 'imagenet':
        if loss == 'crossentropy':
            x_train, y_train, x_test, y_test = data.get_imagenet10()
        else:
            x_train, _, x_test, _ = data.get_imagenet10()
            y_train = np.load('data/' + dataset + '_' + classifier_name + '.npy')
            y_test = np.load('data/' + dataset + '_' + classifier_name + '_test.npy')

    return x_train,y_train, x_test, y_test

def get_generator(dataset):
    if dataset == 'mnist':
        generator= models.get_mnist_autoencoder()
    if dataset == 'cifar10':
        generator = models.get_cifar10_autoencoder()
    if dataset == 'imagenet':
        generator = models.get_imagenet_autoencoder()

    return generator

def get_classifier(dataset,classifier_name):

    if dataset == 'mnist':
        if classifier_name == 'c1':
            classifier = classifiers.get_mnist_classifier_c1()

        if classifier_name == 'c2':
            classifier = classifiers.get_mnist_classifier_c2()

    if dataset == 'cifar10':
        if classifier_name == 'c3':
            classifier = classifiers.get_cifar10_classifier_c3()

        if classifier_name == 'vgg16':
            classifier = classifiers.get_cifar10_vgg16()
    if dataset == 'imagenet':
        if classifier_name == 'vgg16':
            classifier = classifiers.get_imagenet_vgg16()

        if classifier_name == 'mobilenet':
            classifier = classifiers.get_imagenet_mobilenet()

    return classifier



def start_training(dataset,classifier_name,loss,batch_size):
    x_train, y_train, x_test, y_test = get_data(dataset=dataset,classifier_name=classifier_name,loss=loss)
    generator= get_generator(dataset=dataset)
    classifier = get_classifier(dataset=dataset,classifier_name=classifier_name)

    save_path = 'networks/'+dataset+'/' + classifier_name + '_autoencoder_' + loss + '.h5'
    training(x_train, y_train, x_test, y_test, batch_size, generator, classifier, save_path, loss)



if __name__ == '__main__':
    dataset = 'mnist'
    classifier_name = 'c1'
    loss = 'kld'
    batch_size = 512

    start_training(dataset=dataset,classifier_name=classifier_name,loss=loss,batch_size=batch_size)




