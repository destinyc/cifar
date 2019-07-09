from read_data import Read_cifar10
from net import VGG16_Net
from net import VGG_net, Le_net
import tensorflow as tf
import time

def main():
    # read = Read_cifar10()
    # train_data, train_label, test_data, test_label = read.read_data()
    # net = VGG16_Net()
    # net.build()
    #
    # data_batch, label_batch = net.read_batch_data(train_data, train_label)
    # net.train(data_batch, label_batch, model_path = None)

    # net.test(train_data[:100], train_label[:100], model_path = 'model\\vgg16_6000.ckpt')

    net = VGG_net()
    net.build_vgg_16()
    net.train()



main()
















