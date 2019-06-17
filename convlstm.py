from chainer import link, Chain
from chainer import initializers
from chainer import Variable
from chainer import variable
from chainer import reporter
from chainer import links as L
from chainer import functions as F

class ConvLSTM(Chain):
    
    """
    Convolutional LSTM layer with peephole connections.
    
    This is a Convolutional LSTM layer with peephole connections as a chain.
    
    Given an input matrix :math:`X_{t}`, ConvLSTM returns the next hidden matrix :math:`H_{t}` defined as
    
    .. math::
    
        i_{t} = sigma(W_{Xi} * X_{t} + W_{Hi} * H_{t-1} + W_{Ci} ⦿ C_{t-1} + b_{i}),
        f_{t} = sigma(W_{Xf} * X_{t} + W_{Hf} * H_{t-1} + W_{Cf} ⦿ C_{t-1} + b_{f}),
        C_{t} = f_{t} ⦿ C_{t-1} + i_{t} ⦿ tanh(W_{Xc} * X_{t} + W_{Hc} * H_{t-1} + b_{c}),
        o_{t} = sigma(W_{Xo} * X_{t} + W_{Ho} * H_{t-1} + W_{Co} ⦿ C_{t} + b_{o}),
        H_{t} = o_{t} ⦿ tanh(C_{t}),
        
    where sigma is tha sigmoid function, * is the convolution, ⦿ is the Hadamard product,
    C_{t} is the cell state at time t and H_{t} is the hidden matrix at time t.
    
    Args:
        in_channels (int or None): Number of channels of input arrays X_{t}.
        hidden_channels (int): Number of channels of output arrays H_{t}.
        ksize (int or pair of ints): Size of filters (a.k.a kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        nobias (bool): If ``True``, then this link does not use the bias term.
        
    Attributes:
        C_t (~chainer.Variable): Cell states of Convolutional LSTM units.
        H_t (~chainer.Variable): Output at the current time step.

    """
    
    def __init__(self, in_channels, hidden_channels, ksize, nobias=False, **kwargs):
        super(ConvLSTM, self).__init__(

            # input gate
            WXi = L.Convolution2D(in_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2)),
            WHi = L.Convolution2D(hidden_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2), nobias=True),

            # forget gate
            WXf = L.Convolution2D(in_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2)),
            WHf = L.Convolution2D(hidden_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2), nobias=True),

            # update cell
            WXc = L.Convolution2D(in_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2)),
            WHc = L.Convolution2D(hidden_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2), nobias=True),

            # output gate
            WXo = L.Convolution2D(in_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2)),
            WHo = L.Convolution2D(hidden_channels, hidden_channels, ksize, 1, pad=int((ksize - 1) / 2), nobias=True)

        )
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ksize = ksize
        
        self.pc = self.ph = None
            
        with self.init_scope():
            WCi_initializer = initializers.Zero()
            self.WCi = variable.Parameter(WCi_initializer)

            WCf_initializer = initializers.Zero()
            self.WCf = variable.Parameter(WCf_initializer)

            WCo_initializer = initializers.Zero()
            self.WCo = variable.Parameter(WCo_initializer)
    
    def reset_state(self, pc=None, ph=None):
        self.pc = pc
        self.ph = ph
    
    def init_params(self, shape):
        self.WCi.initialize((self.hidden_channels, shape[2], shape[3]))
        self.WCf.initialize((self.hidden_channels, shape[2], shape[3]))
        self.WCo.initialize((self.hidden_channels, shape[2], shape[3]))
    
    def init_state(self, shape):
        self.pc = Variable(self.xp.zeros((shape[0], self.hidden_channels, shape[2], shape[3]), dtype = self.xp.float32))
        self.ph = Variable(self.xp.zeros((shape[0], self.hidden_channels, shape[2], shape[3]), dtype = self.xp.float32))
            
    def forward(self, X):
        if self.WCi.data is None:
            self.init_params(X.data.shape)

        if self.pc is None:
            self.init_state(X.data.shape)
        
        i_t = F.sigmoid(self.WXi(X) + self.WHi(self.ph) + F.scale(self.pc, self.WCi, 1))
        f_t = F.sigmoid(self.WXf(X) + self.WHf(self.ph) + F.scale(self.pc, self.WCf, 1))
        C_t = f_t * self.pc + i_t * F.tanh(self.WXc(X) + self.WHc(self.ph))
        o_t = F.sigmoid(self.WXo(X) + self.WHo(self.ph) + F.scale(C_t, self.WCo, 1))
        H_t = o_t * F.tanh(C_t)

        self.pc = C_t
        self.ph = H_t
        
        return H_t

class Model(Chain):
    
    def __init__(self, n_input, size, **kwargs):
        super(Model, self).__init__(
            # encoding network
            e1 = ConvLSTM(n_input, size[0], ksize=5),
            e2 = ConvLSTM(size[0], size[1], ksize=5),
            e3 = ConvLSTM(size[1], size[2], ksize=5),
            
            # forecasting network
            f1 = ConvLSTM(n_input, size[0], ksize=5),
            f2 = ConvLSTM(size[0], size[1], ksize=5),
            f3 = ConvLSTM(size[1], size[2], ksize=5),
            
            # output
            l7 = L.Convolution2D(sum(size), n_input, ksize=1)
        )
        
        self.n_input = n_input            

    def __call__(self, x, t):
        self.e1.reset_state()
        self.e2.reset_state()
        self.e3.reset_state()

        We = self.xp.array([[i == j for i in range(self.n_input)] for j in range(self.n_input)], dtype=self.xp.float32)
        for i in range(x.shape[1]):     
            xi = F.embed_id(x[:, i, :, :], We)
            xi = F.transpose(xi, (0, 3, 1, 2))
            
            h1 = self.e1(xi)
            h2 = self.e2(h1)
            self.e3(h2)

        self.f1.reset_state(self.e1.pc, self.e1.ph)
        self.f2.reset_state(self.e2.pc, self.e2.ph)
        self.f3.reset_state(self.e3.pc, self.e3.ph)

        loss = None
        
        for i in range(t.shape[1]):
            xs = x.shape
            
            h1 = self.f1(Variable(self.xp.zeros((xs[0], self.n_input, xs[2], xs[3]), dtype=self.xp.float32)))
            h2 = self.f2(h1)
            h3 = self.f3(h2)
            h = F.concat((h1, h2, h3))
            ans = self.l7(h)

            cur_loss = F.softmax_cross_entropy(ans, t[:, i, :, :])
            loss = cur_loss if loss is None else loss + cur_loss
            
        reporter.report({'loss': loss}, self)
        
        return loss
