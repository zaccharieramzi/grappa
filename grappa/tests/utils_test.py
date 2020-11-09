import numpy as np
import tensorflow as tf

from grappa.utils import pinv


def test_pinv():
    a = np.random.normal(size=[10, 10])
    a = a  + 1j * np.random.normal(size=[10, 10])
    a_pinv = np.linalg.pinv(a)
    a_pinv_tf = pinv(tf.constant(a, dtype=tf.complex64))
    np.testing.assert_allclose(a_pinv, a_pinv_tf.numpy(), rtol=1e-5, atol=1e-5)
