import tensorflow as tf
from tensorflow.keras.models import Sequential
from tf_complex.dense import ComplexDense

def linear_deep_grappa_model(ncoils=15):
    model = Sequential([ComplexDense(ncoils, use_bias=False, activation='linear', dtype=tf.complex64)])
    return model
