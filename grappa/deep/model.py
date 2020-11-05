import tensorflow as tf
from tensorflow.keras.models import Model
from tf_complex.dense import ComplexDense


class DeepKSpaceFiller(Model):
    def __init__(
            self,
            ncoils=15,
            n_dense=2,
            instance_normalisation=False,
            kernel_learning=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.ncoils = ncoils
        self.n_dense = n_dense
        self.instance_normalisation = instance_normalisation
        self.kernel_learning = kernel_learning
        n_units = self.ncoils**2 if self.kernel_learning else self.ncoils
        self.denses = [
            ComplexDense(
                n_units,
                use_bias=self.kernel_learning,
                activation='crelu',
            )
            for _ in range(self.n_dense-1)
        ]
        self.denses.append(
            ComplexDense(
                n_units,
                use_bias=self.kernel_learning,
                activation='linear',
            )
        )

    def call(self, inputs):
        if self.instance_normalisation:
            inputs = inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)
            inputs = inputs / tf.reduce_max(tf.abs(inputs), axis=1, keepdims=True)
        outputs = inputs
        for dense in self.denses:
            outputs = dense(outputs)
        if self.kernel_learning:
            kernel = tf.reshape(outputs, [-1, self.ncoils, self.ncoils])
            outputs = kernel @ inputs
        return outputs
