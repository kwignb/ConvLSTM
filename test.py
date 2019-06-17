import numpy as np
import argparse
import chainer
from chainer import iterators, optimizers, serializers
from chainer import cuda
from chainer import Variable
import convlstm
import dataset

def test():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--id', '-i', type=int, default=0)
    parser.add_argument('--inf', type=int, default=10)
    parser.add_argument('--outf', type=int, default=3)
    args = parser.parse_args()

    test = dataset.UCSDped1Dataset(0, 200, args.inf, args.outf, "./ucsd_ped1_test.npy")

    model = convlstm.Model(n_input=2, size=[128,64,64])
    
    if args.model != None:
        print("loading model from " + args.model)
        serializers.load_npz(args.model, model)
    
    x, t = test[args.id]

    x = np.expand_dims(x, 0)
    t = np.expand_dims(t, 0)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()
        x = cuda.cupy.array(x)
        t = cuda.cupy.array(t)

    print(x.shape)
    print(t.shape)

    res = model(Variable(x), Variable(t))
    print(res)

if __name__ == '__main__':
    test()
