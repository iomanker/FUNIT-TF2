import tensorflow as tf
from blocks import *
from layers import *
# Zy
class ClassEncoder(tf.keras.layers.Layer):
    def __init__(self,downs,latent_dim,n_filters,norm,activation,pad_type, **kwargs):
        super(ClassEncoder, self).__init__(**kwargs)
        self.layers = []
        self.layers.append(Conv2DBlock(n_filters, 7, 1, 3,
                                   norm=norm,
                                   activation=activation,
                                   pad_type=pad_type))
        for _ in range(2):
            self.layers.append(Conv2DBlock(2* n_filters, 4, 2, 1,
                                       norm=norm,
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters *= 2
        for _ in range(downs - 2):
            self.layers.append(Conv2DBlock(n_filters, 4, 2, 1,
                                       norm=norm,
                                       activation=activation,
                                       pad_type=pad_type))
        self.layers.append(tf.keras.layers.AveragePooling2D(1))
        self.layers.append(tf.keras.layers.Conv2D(latent_dim, 1, 1, 'valid'))
        self.output_filters = n_filters
    def call(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
# Zx
class ContentEncoder(tf.keras.layers.Layer):
    def __init__(self,downs,n_res,n_filters,norm,activation,pad_type):
        super(ContentEncoder, self).__init__()
        self.layers = []
        self.layers.append(Conv2DBlock(n_filters, 7, 1, 3,
                                   norm=norm,
                                   activation=activation,
                                   pad_type=pad_type))
        for _ in range(downs):
            self.layers.append(Conv2DBlock(2* n_filters, 4, 2, 1,
                                       norm=norm,
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters *= 2
        for _ in range(n_res):
            self.layers.append(ResnetIdentityBlock(n_filters, norm=norm,
                                     activation=activation,
                                     pad_type=pad_type))
        self.output_filters = n_filters
    def call(self,x):
        for l in self.layers:
            x = l(x)
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self,ups,n_res,n_filters,out_dim,activation,pad_type, **kwargs):
        super(Decoder,self).__init__(**kwargs)
        self.AdaIN_layers = []
        # block.model.layers[0].model.layers[0].norm =(address) block.adaIN[0]
        for _ in range(n_res):
            self.AdaIN_layers.append(Res_AdaINBlock(n_filters,
                                                   activation=activation,
                                                   pad_type=pad_type))
        
        # Transposed Conv2D vs Upsampling with Conv
        # https://github.com/keras-team/keras/issues/7307
        # https://distill.pub/2016/deconv-checkerboard/
        self.layers = []
        for _ in range(ups):
            self.layers.append(tf.keras.layers.UpSampling2D(2))
            self.layers.append(Conv2DBlock(n_filters//2, 5, 1, 2,
                                       norm='in',
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters //= 2
        self.layers.append(Conv2DBlock(out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type))
    def call(self,x,y):
        for l in self.AdaIN_layers:
            x, _ = l(x,y)
        for l in self.layers:
            x = l(x)
        return x
    
class MLP(tf.keras.layers.Layer):
    def __init__(self, out_dim, dim, n_blk, activation):
        super(MLP,self).__init__()
        self.layers = []
        self.layers.append(LinearBlock(dim,
                                   activation=activation))
        for _ in range(n_blk - 2):
            self.layers.append(LinearBlock(dim,
                                       activation=activation))
        self.layers.append(LinearBlock(out_dim,
                                   activation='none'))
    def call(self,x):
        for l in self.layers:
            x = l(x)
        return x
