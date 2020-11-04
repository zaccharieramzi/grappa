import tensorflow as tf
from tensorflow.keras.models import Model
from tf_complex.dense import ComplexDense


class DeepKSpaceFiller(Model):
    def __init__(
            self,
            n_coils=15,
            n_dense=2,
            instance_normalisation=False,
            kernel_learning=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_coils = n_coils
        self.n_dense = n_dense
        self.instance_normalisation = instance_normalisation
        self.kernel_learning = kernel_learning
        n_units = self.n_coils**2 if self.kernel_learning else self.n_coils
        self.denses = [
            ComplexDense(
                n_units,
                use_bias=self.kernel_learning,
                activation='crelu',
                dtype=tf.complex64,
            )
            for _ in range(self.n_dense-1)
        ]
        self.denses.append(
            ComplexDense(
                n_units,
                use_bias=self.kernel_learning,
                activation='linear',
                dtype=tf.complex64,
            )
        )

    def call(self, inputs):
        if self.instance_normalisation:
            inputs = inputs - tf.reduce_mean(inputs, axis=1)
            inputs = inputs / tf.reduce_max(tf.abs(inputs), axis=1)
        outputs = inputs
        for dense in self.denses:
            outputs = dense(outputs)
        if self.kernel_learning:
            kernel = tf.reshape(outputs, [-1, self.n_coils, self.n_coils])
            outputs = kernel @ inputs
        return outputs
