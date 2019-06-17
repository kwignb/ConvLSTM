import numpy as np
import argparse
import chainer
from itertools import chain
from chainer import datasets
from chainer import training
from chainer import iterators, optimizers, serializers
from chainer import cuda
from chainer.training import extensions
import convlstm
import dataset

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--validation', '-v', type=int, default=5)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--inf', type=int, default=3)
    parser.add_argument('--outf', type=int, default=3)
    parser.add_argument('--batch', '-b', type=int, default=1)
    args = parser.parse_args()

    train = dataset.UCSDped1Dataset(0, 200, args.inf, args.outf, "./ucsd_ped1_train.npy")

    # cross validation
    dataset_ = datasets.get_cross_validation_datasets(train, args.validation, order=None)
    
    v = 1
    while v <= args.validation:

        model = convlstm.Model(n_input=2, size=[128,64,64])

        if args.model != None:
            print( "loading model from " + args.model )
            serializers.load_npz(args.model, model)

        if args.gpu >= 0:
            cuda.get_device_from_id(0).use()
            model.to_gpu()

        optimizer = optimizers.RMSprop(lr=args.lr)
        optimizer.setup(model)
        
        if args.opt != None:
            print( "loading opt from " + args.opt )
            serializers.load_npz(args.opt, opt)

        train_iter = chainer.iterators.SerialIterator(dataset_[v-1][0], batch_size=args.batch, shuffle=False)
        test_iter = chainer.iterators.SerialIterator(dataset_[v-1][1], batch_size=args.batch, repeat=False, shuffle=False)
        
        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')
        
        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
          
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log_'+str(v)+'_epoch'))
        trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))

        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                       x_key='epoch', file_name='loss_'+str(v)+'_epoch.png'))
        trainer.extend(extensions.ProgressBar(update_interval=1))
        
        trainer.run()
        
        modelname = "./results/model" + str(v)
        print( "saving model to " + modelname )
        serializers.save_npz(modelname, model)

        optname = "./results/opt" + str(v)
        print( "saving opt to " + optname )
        serializers.save_npz(optname, optimizer)

        v = v + 1
if __name__ == '__main__':
    train()
