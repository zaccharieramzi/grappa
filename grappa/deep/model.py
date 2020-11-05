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
            distance_from_center_feat=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.ncoils = ncoils
        self.n_dense = n_dense
        self.instance_normalisation = instance_normalisation
        self.kernel_learning = kernel_learning
        self.distance_from_center_feat = distance_from_center_feat

    def build(self, input_shape):
        if self.kernel_learning:
            n_features = input_shape[-1]
            using_distance_feature = tf.math.mod(n_features, 2)
            n_units = n_features - using_distance_feature
        else:
            n_units = 1
        self.denses = [
            ComplexDense(
                n_units * self.ncoils,
                use_bias=self.kernel_learning,
                activation='crelu',
            )
            for _ in range(self.n_dense-1)
        ]
        self.denses.append(
            ComplexDense(
                n_units * self.ncoils,
                use_bias=self.kernel_learning,
                activation='linear',
            )
        )

    def call(self, inputs):
        if self.instance_normalisation:
            inputs = inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)
            max_val = tf.reduce_max(tf.abs(inputs), axis=1, keepdims=True)
            inputs = inputs / tf.cast(max_val, dtype=inputs.dtype)
        outputs = inputs
        for dense in self.denses:
            outputs = dense(outputs)
        if self.kernel_learning:
            batch_size = tf.shape(inputs)[0]
            kernel = tf.reshape(outputs, [batch_size, -1, self.ncoils])
            if self.distance_from_center_feat:
                inputs = inputs[:, :-1]
            outputs = tf.linalg.matvec(kernel, inputs)
        return outputs
