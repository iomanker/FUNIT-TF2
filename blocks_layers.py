import tensorflow as tf
from tensorflow.python.keras.engine import InputSpec

class AdaptiveInstanceNorm2D(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(AdaptiveInstanceNorm2D, self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def call(self,x,y):
        # x: a content input / y: a style input
        x_mean, x_var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        y_mean, y_var = tf.nn.moments(y, axes=[1,2], keepdims=True)
        x_inv = tf.math.rsqrt(x_var + self.epsilon)
        x_normalized = (x - x_mean) * x_inv
        return y_var * x_normalized + y_mean


# https://www.tensorflow.org/api_docs/python/tf/pad
# https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1,1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        
    def get_output_shape_for(self, s):
        # CHANNELS_LAST
        h_pad, w_pad = self.padding
        return (s[0], s[1]+ 2*h_pad, s[2]+ 2*w_pad, s[3])
    def call(self, x):
        h_pad, w_pad = self.padding
        return tf.pad(x, tf.constant([[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0]]), 'REFLECT')
    
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self,n_filters, norm='bn', activation='relu', pad_type='zero'):
        super(ResnetIdentityBlock, self).__init__()
        self.model = tf.keras.Sequential()
        for _ in range(2):
            self.model.add(Conv2DBlock(n_filters, 3, 1, 1, pad_type, norm=norm, activation=activation, ))
    def call(self,x):
        res = x
        out = self.model(x)
        out += res
        return out
        
class Conv2DBlock(tf.keras.Model):
    def __init__(self, n_filters, ks, st, padding=0, pad_type='zero', use_bias=True, norm="", activation='relu', activation_first=False):
        super(Conv2DBlock,self).__init__()
        
        # Padding
        if pad_type == 'reflect':
            self.pad = ReflectionPadding2D((padding,padding))
        elif pad_type == 'zero':
            self.pad = tf.keras.layers.ZeroPadding2D(padding)
        
        # Normalization
        if norm == 'bn':
            self.norm = tf.keras.layers.BatchNormalization()
        elif norm == 'in':
            assert True, 'this repo is not supported, please install tensorflow-addons at first.'
            # self.norm = tfa.layers.InstanceNormalization()
            self.norm = None
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2D()
        else:
            self.norm = None
            
        # Activation
        self.activation_first = activation_first
        if activation == 'relu':
            self.activation = tf.keras.activations.relu
        elif activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh
        else:
            self.activation = None
        
        self.conv = tf.keras.layers.Conv2D(n_filters, ks, st, 'valid', use_bias=use_bias)
        
    def call(self,x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x