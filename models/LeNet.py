from mxnet import gluon


class LeNet(gluon.nn.HybridBlock):
    def __init__(self, classes=10, feature_size=2, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu')
            self.conv2 = gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu')
            self.maxpool = gluon.nn.MaxPool2D(pool_size=2, strides=2)
            self.flat = gluon.nn.Flatten()
            self.dense1 = gluon.nn.Dense(feature_size)
            self.dense2 = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.flat(x)
        ft = self.dense1(x)
        output = self.dense2(ft)

        return output, ft


def _make_conv_block(block_index, num_chan=32, num_layer=2, stride=1, pad=2):
    out = gluon.nn.HybridSequential(prefix='block_%d_' % block_index)
    with out.name_scope():
        for _ in range(num_layer):
            out.add(gluon.nn.Conv2D(num_chan, kernel_size=5, strides=stride, padding=pad))
            out.add(gluon.nn.LeakyReLU(alpha=0.01))
        out.add(gluon.nn.MaxPool2D())

    return out


class LeNetPlus(gluon.nn.HybridBlock):

    def __init__(self, classes=10, feature_size=2, **kwargs):
        super(LeNetPlus, self).__init__(**kwargs)
        num_chans = [32, 64, 128]
        with self.name_scope():
            self.features = gluon.nn.HybridSequential(prefix='')

            for i, num_chan in enumerate(num_chans):
                self.features.add(_make_conv_block(i, num_chan=num_chan))

            self.features.add(gluon.nn.Dense(feature_size))
            self.features.add(gluon.nn.Flatten())
            self.output = gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        ft = self.features(x)
        output = self.output(ft)
        return output, ft
