from keras.layers import Layer
import keras.backend as K


class GlobalWeightedAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='ones',
                                      trainable=True)
        Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2],

    def call(self, x):

        x = x*self.kernel
        x = K.mean(x, axis=1)

        return x


class GlobalWeightedMaxPooling1D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='ones',
                                      trainable=True)
        Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2],

    def call(self, x):

        x = x*self.kernel
        x = K.max(x, axis=1)

        return x


class GlobalWeightedAveragePooling2D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2], 1),
                                      initializer='ones',
                                      trainable=True)
        Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2],

    def call(self, x):

        x = x*self.kernel
        x = K.mean(x, axis=(1, 2))

        return x


class GlobalWeightedMaxPooling2D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2], 1),
                                      initializer='ones',
                                      trainable=True)
        Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3],

    def call(self, x):

        x = x*self.kernel
        x = K.max(x, axis=(1, 2))

        return x
