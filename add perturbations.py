
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import partial

import models
from basics import *
from tqdm import tqdm
import numpy as np



class encryptor_training():
    def __init__(self,opt,generator_t,generator_n,classifier,eps,dataset):

        self.inputH = opt.imageH
        self.inputW = opt.imageW
        self.channel = opt.channel
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size

        self.generator_n_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)

        self.generator_t = generator_t
        self.classifier = classifier
        self.generator_n = generator_n
        self.eps = eps
        self.dataset = dataset


    def start_training(self,x,x_show,x_test,y_test):
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.batch(self.batch_size)

        for epoch in tqdm(range(self.epochs)):

            for x_batch,_ in zip(x,x):

                logits_loss = self.train_step(x_input=x_batch)


            if epoch%100 == 0:
                print('logits_loss:', logits_loss.numpy())

                self.generator_n.save_weights('networks/'+self.dataset+'/generator_n_'+str(self.eps)+'_'+str(epoch)+'.h5')

                x_gen = self.generator_t.predict(x_show)
                x_perturb = self.generator_n.predict(x_gen)

                x_perturb = tf.clip_by_value(x_perturb, -self.eps, self.eps)

                x_adv = x_gen + x_perturb

                x_perturb = (x_perturb + 1.) / 2.

                x_adv = (x_adv + 1.) / 2.

                save_samples(x_perturb, str(self.eps) + '_' + str(epoch) + '_x_perturb',self.dataset)
                save_samples(x_adv, str(self.eps) + '_' + str(epoch) + '_x_adv',self.dataset)


                x_gen = self.generator_t.predict(x_test, batch_size=256)
                x_adv = tf.clip_by_value(self.generator_n.predict(x_gen, batch_size=256), -self.eps, self.eps) + x_gen
                x_adv = (x_adv + 1.) / 2.
                self.classifier.evaluate(x_adv, y_test, verbose=2, batch_size=256)


    @tf.function
    def train_step(self, x_input):

        with tf.GradientTape() as generator_n_tape:
            gen_images = self.generator_t(x_input,training=False)

            perturbations = self.generator_n(gen_images,training=True)

            adv_images = tf.clip_by_value(perturbations, -self.eps, self.eps) + gen_images


            adv_images = (adv_images + 1.) / 2.
            x_input = (x_input + 1.) / 2.

            adv_logits = self.classifier(adv_images,training=False)
            x_logits = self.classifier(x_input,training=False)

            logits_pair_loss = self.MSE(x_logits,adv_logits)


        generator_n_grad = generator_n_tape.gradient(logits_pair_loss,self.generator_n.trainable_variables)
        self.generator_n_optimizer.apply_gradients(zip(generator_n_grad, self.generator_n.trainable_variables))

        return logits_pair_loss

    def MSE(self,real_image, fake_image):
        loss = tf.reduce_mean(tf.square(real_image - fake_image))
        return loss

if __name__ == '__main__':
    import config
    import data
    import classifiers

    classifier = classifiers.get_mnist_classifier_c1()

    classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    parser = config.get_arguments_mnist()

    conf = parser.parse_args()

    conf.epochs = 1001
    conf.batch_size = 256



    x_train, y_train, x_test, y_test = data.get_mnist()
    classifier.evaluate(x_test, y_test, verbose=2)
    x_train = (x_train * 2.) - 1.
    x_train = x_train.astype(np.float32)


    x_test = (x_test * 2.) - 1.
    x_test = x_test.astype(np.float32)

    x_show = x_test[0:25]

    generator_t = models.get_mnist_generator_GAG()
    generator_n = models.get_mnist_generator_GAG()


    my_encryptor = encryptor_training(opt=conf, generator_t=generator_t,generator_n=generator_n,
                                      classifier=classifier, eps=0.2,dataset='mnist')

    my_encryptor.generator_t.load_weights('networks/mnist/generator_t.h5')


    my_encryptor.start_training(x=x_train,x_show=x_show,x_test=x_test,y_test=y_test)





