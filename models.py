import tensorflow as tf
from blocks import *
from layers import *
# Zy
class ClassEncoder(tf.keras.Model):
    def __init__(self,downs,latent_dim,n_filters,norm,activation,pad_type):
        super(ClassEncoder, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Conv2DBlock(n_filters, 7, 1, 3,
                                   norm=norm,
                                   activation=activation,
                                   pad_type=pad_type))
        for _ in range(2):
            self.model.add(Conv2DBlock(2* n_filters, 4, 2, 1,
                                       norm=norm,
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters *= 2
        for _ in range(downs - 2):
            self.model.add(Conv2DBlock(n_filters, 4, 2, 1,
                                       norm=norm,
                                       activation=activation,
                                       pad_type=pad_type))
        self.model.add(tf.keras.layers.AveragePooling2D(1))
        self.model.add(tf.keras.layers.Conv2D(latent_dim, 1, 1, 'valid'))
        self.output_filters = n_filters
    def call(self, x):
        return self.model(x)
    
# Zx
class ContentEncoder(tf.keras.Model):
    def __init__(self,downs,n_res,n_filters,norm,activation,pad_type):
        super(ContentEncoder, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Conv2DBlock(n_filters, 7, 1, 3,
                                   norm=norm,
                                   activation=activation,
                                   pad_type=pad_type))
        for _ in range(downs):
            self.model.add(Conv2DBlock(2* n_filters, 4, 2, 1,
                                       norm=norm,
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters *= 2
        for _ in range(n_res):
            self.model.add(ResnetIdentityBlock(n_filters, norm=norm,
                                     activation=activation,
                                     pad_type=pad_type))
        self.output_filters = n_filters
    def call(self,x):
        return self.model(x)
    
class Decoder(tf.keras.Model):
    def __init__(self,ups,n_res,n_filters,out_dim,activation,pad_type):
        super(Decoder,self).__init__()
        self.AdaIN_model = []
        # block.model.layers[0].model.layers[0].norm =(address) block.adaIN[0]
        for _ in range(n_res):
            self.AdaIN_model.append(Res_AdaINBlock(n_filters,
                                                   activation=activation,
                                                   pad_type=pad_type))
        
        # Transposed Conv2D vs Upsampling with Conv
        # https://github.com/keras-team/keras/issues/7307
        # https://distill.pub/2016/deconv-checkerboard/
        self.model = tf.keras.Sequential()
        for _ in range(ups):
            self.model.add(tf.keras.layers.UpSampling2D(2))
            self.model.add(Conv2DBlock(n_filters//2, 5, 1, 2,
                                       norm='in',
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters //= 2
        self.model.add(Conv2DBlock(out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type))
    def call(self,x,y):
        for layer in self.AdaIN_model:
            x, _ = layer(x,y)
        return self.model(x)
    
class MLP(tf.keras.Model):
    def __init__(self, out_dim, dim, n_blk, activation):
        super(MLP,self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(LinearBlock(dim,
                                   activation=activation))
        for _ in range(n_blk - 2):
            self.model.add(LinearBlock(dim,
                                       activation=activation))
        self.model.add(LinearBlock(out_dim,
                                   activation='none'))
    def call(self,x):
        return self.model(x)