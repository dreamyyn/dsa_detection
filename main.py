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
from model import patch_block_classifier

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', default='0', help='0,1,2,3')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='../data/DSA_patch', help='name of the dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='../checkpoint/patch_wise', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='../log/patch_wise', help='logs are saved here')
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--validation_split', dest='validation_split', type=float, default=0.1, help='random split validation set from training dataset')
parser.add_argument('--input_size', dest='input_size', type=int, default=128, help='resize input image size')
# parser.add_argument('--crop_size', dest='crop_size', type=int, default=224, help='crop image size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=20, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', action="store_true", help='if continue training, load the latest model')
parser.set_defaults(continue_train=False)
parser.add_argument('--frame_num', dest='frame_num', type=int, default=7, help='frame number for each patch')
parser.add_argument('--cls_num', dest='cls_num', type=int, default=2, help='positive / negative')

args = parser.parse_args()

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = patch_block_classifier(sess, epochs=args.epochs, validation_split=args.validation_split,
                        dataset_dir=args.dataset_dir, log_dir=args.log_dir, checkpoint_dir=args.checkpoint_dir,
                        batch_size=args.batch_size, input_size=args.input_size, lr=args.lr, beta1=args.beta1,
                        print_freq=args.print_freq, continue_train=args.continue_train, phase=args.phase,
                        frame_num=args.frame_num, cls_num=args.cls_num)

        if args.phase == 'train':
            model.train()
        else:
            model.test()

if __name__ == '__main__':
    tf.app.run()
