import tensorflow as tf

# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py#L161-L185
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5, affine=False):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.affine = affine

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True) if self.affine else None

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True) if self.affine else None

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return normalized if self.affine == False else self.scale * normalized + self.offset

class AdaptiveInstanceNorm2D(tf.keras.layers.Layer):
    def __init__(self, num_features, epsilon=1e-5, **kwargs):
        super(AdaptiveInstanceNorm2D, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.num_features = num_features
    
    def call(self,x,y):
        # x: a content input / y: a style input
        x_mean, x_var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        y_shape = tf.shape(y)
        
        # y =  tf.reshape(y,[int(y_shape[0]),1,1,int(y_shape[1])])
        # y = tf.reduce_mean(y,[1,2],keepdims=True)
        y_mean, y_var = tf.split(y,num_or_size_splits=2,axis=1)
        y_mean = y_mean[:, tf.newaxis, tf.newaxis, :]
        y_var = y_var[:, tf.newaxis, tf.newaxis, :]
        x_inv = tf.math.rsqrt(x_var + self.epsilon)
        x_normalized = (x - x_mean) * x_inv
        return y_var * x_normalized + y_mean


# https://www.tensorflow.org/api_docs/python/tf/pad
# https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1,1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        
    def get_output_shape_for(self, s):
        # CHANNELS_LAST
        h_pad, w_pad = self.padding
        return (s[0], s[1]+ 2*h_pad, s[2]+ 2*w_pad, s[3])
    def call(self, x):
        h_pad, w_pad = self.padding
        return tf.pad(x, tf.constant([[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0]]), 'REFLECT')