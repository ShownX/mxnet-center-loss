from mxnet import gluon
import numpy as np


class CenterLoss(gluon.HybridBlock):
    r"""
    Center Loss: A Discriminative Feature Learning Approach for Deep Face Recognition
    """
    def __init__(self, num_classes, feature_size, lmbd, **kwargs):
        super(CenterLoss, self).__init__(**kwargs)
        self._num_classes = num_classes
        self._feature_size = feature_size
        self._lmda = lmbd
        self.centers = self.params.get('centers', shape=(num_classes, feature_size))

    def hybrid_forward(self, F, feature, label, centers):

        hist = F.array(np.bincount(label.asnumpy().astype(int)))

        centers_count = F.take(hist, label)

        centers_selected = F.take(centers, label)

        diff = feature - centers_selected

        loss = self._lmda * 0.5 * F.sum(F.square(diff), 1) / centers_count

        return F.mean(loss, axis=0, exclude=True)
