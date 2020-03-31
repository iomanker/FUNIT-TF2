import tensorflow as tf
from tensorflow.python.keras.engine import InputSpec

# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
class VectorQuantizer(tf.keras.layers.Layer):
    """Neural Discrete Representation Learning (https://arxiv.org/abs/1711.00937)"""
    
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, name='vq_layer'):
        super(VectorQuantizer, self).__init__(name=name)
        # embedding_dim: D, num_embeddings: K
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        with tf.init_scope():
            initializer = tf.keras.initializers.GlorotUniform()
            # (D, K)
            self._w = self.add_weight('embedding', shape=[embedding_dim, num_embeddings],
                                    initializer=initializer, trainable=True)
            
        
    def call(self, inputs, training=True):
        # (B,H,W,D)
        input_shape = tf.shape(inputs)
        # with tf.control_dependencies(...)
        # (BxHxW, D)
        flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])
        
        # (BxHxW, K) = (BxHxW, 1) - (BxHxW, D) x (D, K) + (1, K)
        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True))\
                    - 2 * tf.matmul(flat_inputs, self._w)\
                    + tf.reduce_sum(self._w**2, 0, keepdims=True)
        
        encoding_indices = tf.argmax(-distances, 1) # (BxHxW)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings) # (BxHxW, K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1]) # (B, H, W)
        quantized = self.quantize(encoding_indices) # NOTICE (B, H, W, D)
        
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # WHY?
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, 0)
        # It indicates how many codes are 'active' on average.
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        return {'quantize': quantized,
                'loss': loss,
                'perplexity': perplexity,
                'encodings': encodings,
                'encoding_indices': encoding_indices}
    
    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices): # (B, H, W)
        with tf.control_dependencies([encoding_indices]):
            w = tf.transpose(self.embeddings.read_value(), [1,0]) # (K, D)
        return tf.nn.embedding_lookup(w, encoding_indices)  # (B, H, W, D)

# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py#L161-L185
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class AdaptiveInstanceNorm2D(tf.keras.layers.Layer):
    def __init__(self, num_features, epsilon=1e-6, **kwargs):
        super(AdaptiveInstanceNorm2D, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.num_features = num_features
    
    def call(self,x,y):
        # x: a content input / y: a style input
        x_mean, x_var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        y_shape = tf.shape(y)
        
        # y =  tf.reshape(y,[int(y_shape[0]),1,1,int(y_shape[1])])
        y = tf.reduce_mean(y,[1,2],keepdims=True)
        y_mean, y_var = tf.split(y,num_or_size_splits=2,axis=3)
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