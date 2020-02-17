import tensorflow as tf

class AdaptiveInstanceNorm2D(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(AdaptiveInstanceNorm2D, self).__init__()
        self.epsilon = epsilon
    
    def call(self,x,y):
        # x: a content input / y: a style input
        x_mean, x_var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        y_mean, y_var = tf.nn.moments(y, axes=[1,2], keepdims=True)
        x_inv = tf.math.rsqrt(x_var + self.epsilon)
        x_normalized = (x - x_mean) * x_inv
        return y_var * x_normalized + y_mean