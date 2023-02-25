import argparse

def get_arguments_mnist():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageH', type=int, default=28, help='the hight of image, default = 28')
    parser.add_argument('--imageW', type=int, default=28, help='the width of image, default = 28')
    parser.add_argument('--channel', type=int, default=1, help='the channel of image, default=1')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs, default = 200')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size, default=1024')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    return parser

def get_arguments_cifar10():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageH', type=int, default=32, help='the hight of image, default = 32')
    parser.add_argument('--imageW', type=int, default=32, help='the width of image, default = 32')
    parser.add_argument('--channel', type=int, default=3, help='the channel of image, default=3')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs, default = 200')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size, default=512')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    return parser


def get_arguments_imagenet():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageH', type=int, default=224, help='the hight of image, default = 224')
    parser.add_argument('--imageW', type=int, default=224, help='the width of image, default = 224')
    parser.add_argument('--channel', type=int, default=3, help='the channel of image, default=3')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs, default = 200')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size, default=16')


    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    return parser