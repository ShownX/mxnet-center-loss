"""
This script is used to run experiment on MNIST using softmax and center loss
"""
import argparse
import os
import timeit
import logging

import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd

from utils import plot_features, evaluate_accuracy, data_loader
from center_loss import CenterLoss
from models import LeNetPlus


def train():
    print('Start to train...')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus != '-1' else mx.cpu()
    print('Loading the data...')

    train_iter, test_iter = data_loader(args.batch_size)

    model = LeNetPlus()
    model.hybridize()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(model.collect_params(),
                            optimizer='sgd', optimizer_params={'learning_rate': args.lr, 'wd': args.wd})

    if args.center_loss:
        center_loss = CenterLoss(args.num_classes, feature_size=2, lmbd=args.lmbd)
        center_loss.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        trainer_center = gluon.Trainer(center_loss.collect_params(),
                                       optimizer='sgd', optimizer_params={'learning_rate': args.alpha})
    else:
        center_loss, trainer_center = None, None

    smoothing_constant, moving_loss = .01, 0.0

    best_acc = 0
    for e in range(args.epochs):
        start_time = timeit.default_timer()

        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(ctx[0])
            label = label.as_in_context(ctx[0])
            with autograd.record():
                output, features = model(data)
                loss_softmax = softmax_cross_entropy(output, label)
                if args.center_loss:
                    loss_center = center_loss(features, label)
                    loss = loss_softmax + loss_center
                else:
                    loss = loss_softmax
            loss.backward()
            trainer.step(data.shape[0])
            if args.center_loss:
                trainer_center.step(data.shape[0])

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        elapsed_time = timeit.default_timer() - start_time

        train_accuracy, train_ft, _, train_lb = evaluate_accuracy(train_iter, model, ctx)
        test_accuracy, test_ft, _, test_lb = evaluate_accuracy(test_iter, model, ctx)

        if args.plotting:
            plot_features(train_ft, train_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, '%s-train-epoch-%s.png' % (args.prefix, e)))
            plot_features(test_ft, test_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, '%s-test-epoch-%s.png' % (args.prefix, e)))

        logging.warning("Epoch [%d]: Loss=%f" % (e, moving_loss))
        logging.warning("Epoch [%d]: Train-Acc=%f" % (e, train_accuracy))
        logging.warning("Epoch [%d]: Test-Acc=%f" % (e, test_accuracy))
        logging.warning("Epoch [%d]: Elapsed-time=%f" % (e, elapsed_time))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            model.save_params(os.path.join(args.ckpt_dir, args.prefix + '-best.params'))


def test():
    print('Start to test...')
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus != '-1' else mx.cpu()

    _, test_iter = data_loader(args.batch_size)

    model = LeNetPlus()
    model.load_params(os.path.join(args.ckpt_dir, args.prefix + '-best.params'), ctx=ctx)

    start_time = timeit.default_timer()
    test_accuracy, features, predictions, labels = evaluate_accuracy(test_iter, model, ctx)
    elapsed_time = timeit.default_timer() - start_time

    print("Test_acc: %s, Elapsed_time: %f s" % (test_accuracy, elapsed_time))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.plotting:
        plot_features(features, labels, num_classes=args.num_classes,
                      fpath=os.path.join(args.out_dir, '%s.png' % args.prefix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convolutional Neural Networks')
    # File related
    parser.add_argument('--prefix', default='softmax', type=str, help='prefix')
    parser.add_argument('--ckpt_dir', default='ckpt', type=str, help='ckpt directory')
    # Training related
    parser.add_argument('--gpus', default='0', type=str, help='gpus')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lmbd', default=1, type=float, help='lambda in the paper')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in the paper')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--center_loss', action='store_true', help='train using center loss')
    parser.add_argument('--plotting', action='store_true', help='generate figure')
    # Test related
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--out_dir', default='output', type=str, help='output dir')

    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()
