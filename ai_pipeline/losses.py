"""
Loss Functions for Retinal Classification.

Implements Categorical Focal Loss to prevent dominant majority classes
(Normal retinas) from drowning out minority disease gradients (Glaucoma, Cataract),
preventing precision collapse and restoring balanced recall.
"""

import tensorflow as tf
try:
    import keras
    from keras import saving as keras_saving
    register_serializable = keras_saving.register_keras_serializable
except (ImportError, AttributeError):
    from tensorflow import keras
    register_serializable = getattr(getattr(keras, "saving", None), "register_keras_serializable", lambda **kw: (lambda c: c))


@register_serializable(package="RetinalClassifier")
class CategoricalFocalLoss(keras.losses.Loss):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = None,
        label_smoothing: float = 0.05,
        name: str = "categorical_focal_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.label_smoothing > 0.0:
            num_cls = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_cls)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        modulating = tf.math.pow(1.0 - y_pred, self.gamma)
        loss = modulating * ce

        if self.alpha is not None:
            loss = loss * tf.constant(self.alpha, dtype=tf.float32)

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing
        })
        return config
