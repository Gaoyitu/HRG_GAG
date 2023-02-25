
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from basics import *
from tqdm import tqdm
import numpy as np



class encryptor_training():
    def __init__(self,opt,generator,discriminator,dataset):

        self.inputH = opt.imageH
        self.inputW = opt.imageW
        self.channel = opt.channel
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.generator_optimizer = tf.keras.optimizers.Adam(0.002,beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.002,beta_1=0.5)
        self.SLoss = tf.keras.losses.BinaryCrossentropy()
        self.CLoss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset


    def start_training(self,x,x_target,y,x_show):
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.batch(self.batch_size)

        x_target = tf.data.Dataset.from_tensor_slices(x_target)
        x_target = x_target.batch(self.batch_size)

        y = tf.data.Dataset.from_tensor_slices(y)
        y = y.batch(self.batch_size)


        for epoch in tqdm(range(self.epochs)):

            for x_batch,x_t_batch,y_batch in zip(x,x_target,y):

                discr_loss,gen_loss= self.train_step(x_input=x_batch,x_target=x_t_batch,label=y_batch)



            if epoch%100 == 0:
                print('d_loss:', discr_loss.numpy())
                print('g_loss:', gen_loss.numpy())
                self.generator.save_weights('networks/'+self.dataset+'/generator_t_'+str(epoch)+'.h5')
                self.discriminator.save_weights('networks/'+self.dataset+'/discriminator_'+str(epoch)+'.h5')

                x_gen = self.generator.predict(x_show)
                x_gen = (x_gen + 1.) / 2.
                save_samples(x_gen, str(epoch) + '_gen',self.dataset)


    @tf.function
    def train_step(self, x_input,x_target,label):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
            gen_images = self.generator(x_input,training=True)

            fake_S_out, fake_C_out = self.discriminator(gen_images,training=True)
            real_S_out, real_C_out = self.discriminator(x_target,training=True)


            discr_loss = self.discriminator_loss(real_S_out=real_S_out,
                                                          real_C_out=real_C_out,
                                                          fake_S_out=fake_S_out,
                                                          fake_C_out=fake_C_out,
                                                          label=label)
            gen_loss =self.generator_loss(fake_S_out=fake_S_out,
                                          fake_C_out=fake_C_out,
                                          label=label)

        gen_grad = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        discr_grad = discr_tape.gradient(discr_loss,self.discriminator.trainable_variables)


        self.generator_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discr_grad,self.discriminator.trainable_variables))

        return discr_loss,gen_loss


    def discriminator_loss(self,real_S_out,real_C_out, fake_S_out, fake_C_out, label):
        real_loss = self.SLoss(tf.ones_like(real_S_out),real_S_out)
        real_class_loss = self.CLoss(label,real_C_out)
        fake_loss = self.SLoss(tf.zeros_like(fake_S_out),fake_S_out)
        fake_class_loss = self.CLoss(label,fake_C_out)

        return real_loss+real_class_loss+fake_loss+fake_class_loss

    def generator_loss(self,fake_S_out,fake_C_out, label):
        fake_loss = self.SLoss(tf.ones_like(fake_S_out),fake_S_out)
        fake_class_loss = self.CLoss(label,fake_C_out)

        return fake_loss+fake_class_loss




if __name__ == '__main__':


    import config
    import data
    import models

    parser = config.get_arguments_mnist()

    conf = parser.parse_args()


    conf.epochs = 10000
    conf.batch_size = 256

    dataset = 'mnist'


    generator = models.get_mnist_generator_GAG()
    discriminator = models.get_mnist_discriminator()


    my_encryptor = encryptor_training(opt=conf,generator=generator,discriminator=discriminator,dataset=dataset)

    x_train = np.load('data/mnist/x_train.npy')
    x_target = np.load('data/mnist/x_train_target.npy')
    y_train = np.load('data/mnist/y_train.npy')

    x_train = (x_train*2.) - 1.

    x_target = (x_target * 2.) - 1.

    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(x_target)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    x_train = x_train.astype(np.float32)

    x_target = x_target.astype(np.float32)

    _, _, x_test, y_test = data.get_mnist()
    x_test = (x_test * 2.) - 1.
    x_test = x_test.astype(np.float32)

    x_show = x_test[0:25]
    my_encryptor.start_training(x=x_train,x_target=x_target,y=y_train,x_show=x_show)





