import tensorflow as tf
import matplotlib.pyplot as plt

class AdamOptWrapper(tf.keras.optimizers.Adam):
    def __init__(self,
                 learning_rate=1e-4,
                 beta_1=0.5,
                 beta_2=0.999,
                 epsilon=1e-4,
                 amsgrad=False,
                 **kwargs):
        super(AdamOptWrapper, self).__init__(learning_rate, beta_1, beta_2, epsilon,
                                             amsgrad, **kwargs)

def show_gray_image(image):
    plt.imshow(image,cmap="gray")
    plt.axis("off")
    plt.show()

def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def save_samples(images,file_name,dataset):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=5, ncols=5, sharex=True, sharey=True)


    for ax, img in zip(axes.flatten(), images):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if dataset == 'mnist':
            im = ax.imshow(img, cmap='gray')
        else:
            im = ax.imshow(img)
    f = plt.gcf()
    f.savefig(r'images//'+dataset+'//{}.png'.format(file_name))
    f.clear()

    return fig, axes


