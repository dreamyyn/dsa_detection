from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pdb
from six.moves import xrange
import argparse
from ops import *
from util import *


class patch_block_classifier(object):
    def __init__(self, sess, dataset_dir='data/', log_dir='log/',
                    checkpoint_dir='checkpoint/', epochs=100, validation_split=0.1,
                    batch_size=2, input_size=128, lr=0.0002, beta1=0.5,
                    print_freq=100, continue_train=False, phase='train',
                    frame_num=7, cls_num=2):
        self.sess = sess
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_dir = dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.validation_split = validation_split
        self.input_size = input_size
        self.lr = lr
        self.beta1 = beta1
        self.print_freq = print_freq
        self.continue_train = continue_train
        self.phase = phase
        self.frame_num = frame_num
        self.cls_num = cls_num

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.build_model()

    def build_model(self):
        # image and label
        self.inputs = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.frame_num], name='input')
        self.labels = tf.placeholder(tf.float32, [None, 1])

        # loss
        self.logits, self.logits_sig = self.classifier(self.inputs)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.l2_loss = tf.nn.l2_loss(self.logits_sig - self.labels)

        # accuracy
        correct_predict = tf.equal(tf.round(self.logits_sig), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        self.logits_sample, self.logits_sig_sample = self.classifier(self.inputs, reuse=True)
        self.loss_sample = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits_sample))
        # accuracy
        correct_predict = tf.equal(tf.round(self.logits_sig_sample), self.labels)
        self.accuracy_sample = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        # summary
        self.loss_sum = tf.summary.scalar("training loss", self.loss)
        self.acc_sum = tf.summary.scalar("training accuracy", self.accuracy)

        # variables
        self.t_vars = tf.trainable_variables()

        # save model
        self.saver = tf.train.Saver(max_to_keep=5)
        print('finish building tici score predictor')


    def train(self):
        # optimizer
        # self.phase = 'train'

        # optimize the cross entropy
        # optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
        #                   .minimize(self.loss, var_list=self.t_vars)
        # optimize the l2
        optim = tf.train.GradientDescentOptimizer(self.lr).minimize(self.l2_loss, var_list=self.t_vars)
        # optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.l2_loss, var_list=self.t_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # summary writer
        file_path = self.log_dir
        self.train_sum = tf.summary.merge([self.loss_sum, self.acc_sum])
        self.writer = tf.summary.FileWriter(file_path, self.sess.graph)

        # load model
        if self.continue_train == True and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load all data
        data = glob('{}/train/*.{}'.format(self.dataset_dir, 'npz'))
        # np.random.shuffle(data)
        training_data_num = int((1 - self.validation_split) * len(data))
        training_data = data[:training_data_num]
        validation_data = data[training_data_num:]
        batch_idxs = len(training_data) // self.batch_size


        counter = 0
        start_time = time.time()
        acc_best = 0
        # self.validate(validation_data)
        for epoch in xrange(self.epochs):
            np.random.shuffle(training_data)
            for idx in xrange(0, batch_idxs):
                # load data
                batch_files = training_data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                # pdb.set_trace()
                images = [b[0] for b in batch]
                labels = [b[1] for b in batch]
                # if idx > 262:
                #     pdb.set_trace()
                images = np.array(images).astype(np.float32)

                labels = np.array(labels)
                labels = np.reshape(labels, [self.batch_size, 1])

                # pdb.set_trace()
                _, summary_str = self.sess.run([optim, self.train_sum], \
                                    feed_dict={self.inputs:images, self.labels:labels})

                counter += 1
                if counter % self.print_freq == 1:
                    loss = self.loss.eval(feed_dict={self.inputs:images, self.labels:labels})
                    acc = self.accuracy.eval(feed_dict={self.inputs:images, self.labels:labels})

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, acc: %.8f" % (epoch, idx, batch_idxs,
                            time.time() - start_time, loss, acc))
                    self.writer.add_summary(summary_str, counter)

            #validate at the end of each epoch
            # print('validation split')
            # loss_avg, acc_avg = self.validate(validation_data)
            print('validate dir sort')
            loss_avg, acc_avg = self.validate_dir()
            if acc_best < acc_avg: # getting better model, save
                print("save best model!")
                self.save(self.checkpoint_dir, counter, is_best=True)
                acc_best = acc_avg
            elif epoch % 30 == 0:
                self.save(self.checkpoint_dir, counter)


    def classifier(self, image, reuse=False):
        filter_num = [32, 32, 64, 64]
        # filter_num = [32, 32, 64, 64, 128]
        # filter_num = [64, 64, 128, 256, 512]
        kernel_size = [7, 3, 3, 3, 3]
        stride = [2, 1, 2, 2, 2]

        if self.phase == 'train' and reuse == False:
            is_training = True
        else:
            is_training = False

        with tf.variable_scope("classifier") as scope:
            # assert tf.get_variable_scope().reuse == False
            if reuse == True:
                scope.reuse_variables()
            # ResNet-18
            # conv1, [128,128,1]->[64,64,32]
            # with tf.variable_scope('conv1'):
            #     x = conv2d(image, filter_num[0], k_h=kernel_size[0], k_w=kernel_size[0],
            #                 d_h=stride[0], d_w=stride[0], name='conv_1')
            #     x = bn(x, name='bn_1', is_training=is_training)
            #     x = lrelu(x, name='lrelu_1')
            #     # x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], 'SAME')
            #     conv1 = x
            #
            # # conv2, [64,64,32]->[64,64,32]
            # with tf.variable_scope('conv2_1'):
            #     x = residual_block(x, is_training=is_training)
            # with tf.variable_scope('conv2_2'):
            #     x = residual_block(x, is_training=is_training)
            #     conv2 = x
            #
            # # conv3, [64,64,32]->[32,32,64]
            # with tf.variable_scope('conv3_1'):
            #     x = residual_block(x, output_dim=filter_num[2], stride=stride[2], is_first=True, is_training=is_training)
            # with tf.variable_scope('conv3_2'):
            #     x = residual_block(x, is_training=is_training)
            # conv3 = x
            #
            # # conv4, [32,32,64]->[16,16,128]
            # with tf.variable_scope('conv4_1'):
            #     x = residual_block(x, output_dim=filter_num[3], stride=stride[3], is_first=True, is_training=is_training)
            # with tf.variable_scope('conv4_2'):
            #     x = residual_block(x, is_training=is_training)
            # conv4 = x
            #
            # # conv5, [16,16,128]->[8,8,256]
            # with tf.variable_scope('conv5_1'):
            #     x = residual_block(x, output_dim=filter_num[4], stride=stride[4], is_first=True, is_training=is_training)
            # with tf.variable_scope('conv5_2'):
            #     x = residual_block(x, is_training=is_training)
            # conv5 = x
            #
            # # logits
            # with tf.variable_scope('logits'):
            #     x = tf.reduce_mean(x, [1, 2])
            #     x = linear(x, 1)
                            # conv1
            x = conv2d(image, filter_num[0], name='conv_1')
            x = bn(x, name='bn_1', is_training=is_training)
            # x = tf.nn.dropout(x, 0.5)
            x = lrelu(x, name='lrelu_1')
            conv1 = x
            # conv2
            x = conv2d(x, filter_num[1], name='conv_2')
            x = bn(x, name='bn_2', is_training=is_training)
            x = tf.nn.dropout(x, 0.5)
            x = lrelu(x, name='lrelu_2')
            conv2 = x
            # conv3
            x = conv2d(x, filter_num[2], name='conv_3')
            x = bn(x, name='bn_3', is_training=is_training)
            x = tf.nn.dropout(x, 0.5)
            x = lrelu(x, name='lrelu_3')
            conv3 = x
            # conv4
            x = conv2d(x, filter_num[3], name='conv_4')
            x = bn(x, name='bn_4', is_training=is_training)
            x = tf.nn.dropout(x, 0.5)
            x = lrelu(x, name='lrelu_4')

            x = tf.reduce_mean(x, [1, 2])
            x = linear(x, self.cls_num-1)

        return x, tf.nn.sigmoid(x)


    def validate(self, sample_files):
        # load validation input
        print("Loading validation images ...")
        # sample_files = sorted(sample_files)
        sample = [load_data(sample_file) for sample_file in sample_files]
        labels = [b[1] for b in sample]
        sample = [b[0] for b in sample]
        sample = np.array(sample).astype(np.float32)

        sample_images = [sample[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]
        sample_images = np.array(sample_images)
        sample_labels = [labels[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]

        loss_counter = 0
        acc_counter = 0
        # pdb.set_trace()
        for i, sample_image in enumerate(sample_images):
            labels = sample_labels[i]
            idx = i+1
            # if sample_image.shape[0] < self.batch_size:
            labels = np.reshape(np.array(labels), [-1, self.cls_num-1])
            [loss, acc] = self.sess.run([self.loss_sample,self.accuracy_sample], feed_dict={self.inputs:sample_image, self.labels:labels})
            loss_counter = loss + loss_counter
            acc_counter = acc + acc_counter
        loss_avg = loss_counter / idx
        acc_avg = acc_counter / idx
        print('Loss: {}, Accuracy: {}'.format(loss_avg, acc_avg))
        return loss_avg, acc_avg

    # testing version
    def validate_dir(self):
        # load data
        print("Loading validation images ...")
        sample_files = glob('{}/test/*.{}'.format(self.dataset_dir, 'npz'))
        # test overfitting
        # sample_files = glob('{}/train/*.{}'.format(self.dataset_dir, 'npz'))
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.npz')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        sample = [load_data(sample_file) for sample_file in sample_files]
        labels = [b[1] for b in sample]
        sample = [b[0] for b in sample]
        sample = np.array(sample).astype(np.float32)

        sample_images = [sample[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]
        sample_images = np.array(sample_images)
        sample_labels = [labels[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]

        loss_counter = 0
        acc_counter = 0
        label_list = []
        for i, sample_image in enumerate(sample_images):
            # if sample_image.shape[0] < self.batch_size:
            #     break
            idx = i+1
            labels = sample_labels[i]
            labels = np.array(labels)
            labels = np.reshape(labels, [-1, self.cls_num-1])
            # pdb.set_trace()
            loss, acc = self.sess.run([self.loss_sample, self.accuracy_sample], feed_dict={self.inputs:sample_image, self.labels:labels})
            loss_counter = loss + loss_counter
            acc_counter = acc + acc_counter
            label_list.append(labels)
        loss_avg = loss_counter / idx
        acc_avg = acc_counter / idx
        print('average cross entropy loss: ', loss_avg)
        print('average accuracy: ', acc_avg)
        return loss_avg, acc_avg


    def save(self, checkpoint_dir, step, is_best=False):
        model_name = "patch_wise_block.model"
        if is_best == True:
            model_name = 'best.' + model_name
        print("save model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self):
        """Test classifier"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load ckpt
        if self.load(self.checkpoint_dir):
            print(" [*] Load tici score classifier SUCCESS")
        else:
            print(" [!] Load tici score classifier failed...")

        # load data
        sample_files = glob('{}/test/*.{}'.format(self.dataset_dir, 'npz'))
        # test overfitting
        # sample_files = glob('{}/train/*.{}'.format(self.dataset_dir, 'npz'))
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.npz')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        print("Loading testing images ...")
        sample = [load_data(sample_file) for sample_file in sample_files]
        labels = [b[1] for b in sample]
        sample = [b[0] for b in sample]
        sample = np.array(sample).astype(np.float32)

        sample_images = [sample[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]
        sample_images = np.array(sample_images)
        sample_labels = [labels[i:i+self.batch_size]
                         for i in xrange(0, len(sample), self.batch_size)]

        loss_counter = 0
        acc_counter = 0
        label_list = []
        output_list = []
        for i, sample_image in enumerate(sample_images):
            # if sample_image.shape[0] < self.batch_size:
            #     break
            idx = i+1
            labels = sample_labels[i]
            labels = np.array(labels)
            labels = np.reshape(labels, [-1, self.cls_num-1])
            # pdb.set_trace()
            loss, acc, output = self.sess.run([self.loss, self.accuracy, self.logits_sig], feed_dict={self.inputs:sample_image, self.labels:labels})
            loss_counter = loss + loss_counter
            acc_counter = acc + acc_counter
            label_list.append(labels)
            output_list.append(output)

        loss_avg = loss_counter / idx
        acc_avg = acc_counter / idx
        print('average cross entropy loss: ', loss_avg)
        print('average accuracy: ', acc_avg)

        # save result
        label_list = np.concatenate(label_list, axis=0)
        output_list = np.concatenate(output_list, axis=0)
        # with open('classification_test.npz','w') as file_input:
        #     np.savez_compressed(file_input, label=label_list, output=output_list)

        res = np.concatenate([label_list, output_list], axis=1)
        np.savetxt('classification_test.txt', res, fmt='%3f', delimiter='   ')

        # Accuracy
        acc = np.mean((label_list == np.round(output_list)))
        print('testing accuracy: ', acc)
