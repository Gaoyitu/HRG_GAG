import models
import classifiers
import data
from basics import show_gray_image, show_image

def test(dataset,x,y,classifier,generator,model_name):
    methods = ['crossentropy', 'mse', 'kld']
    for method in methods:
        print('method:', method)

        network = 'networks/' + dataset + '/' + model_name + '_generator_' + method + '.h5'

        generator.load_weights(network)
        x_enc = generator.predict(x)
        classifier.evaluate(x_enc, y, verbose=2)


def test_mnist():
    _, _, x_test, y_test = data.get_mnist() #get testing data
    dataset = 'mnist'

    model_name = 'c1'

    classifier = classifiers.get_mnist_classifier_c1() # get classifier
    classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    print('original:')
    classifier.evaluate(x_test, y_test, verbose=2) # evaluate original data
    generator = models.get_mnist_generator_HRG() # get generator


    test(dataset, x_test, y_test, classifier, generator, model_name)





if __name__ =='__main__':

    test_mnist()







