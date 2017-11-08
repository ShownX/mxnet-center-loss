import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as PathEffects
import mxnet as mx
from mxnet import nd


def plot_features(features, labels, num_classes, fpath='features.png'):
    name_dict = dict()
    for i in range(num_classes):
        name_dict[i] = str(i)

    f = plt.figure(figsize=(16, 12))

    palette = np.array(sns.color_palette("hls", num_classes))

    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    f.savefig(fpath)
    plt.close()


def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()

    features, predicts, labels = [], [], []
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])

        output, fts = net(data)

        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)

        features.extend(fts.asnumpy())
        predicts.extend(predictions.asnumpy())
        labels.extend(label.asnumpy())

    features = np.array(features)
    predicts = np.array(predicts)
    labels = np.array(labels)

    return acc.get()[1], features, predicts, labels


def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)


def data_loader(batch_size):
    train_iter = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_iter = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)
    return train_iter, test_iter
