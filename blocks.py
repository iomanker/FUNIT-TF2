import tensorflow as tf
from layers import *

class Conv2DBlock(tf.keras.Model):
    def __init__(self, n_filters, ks, st, padding=0, pad_type='zero', use_bias=True, norm="", activation='relu', activation_first=False):
        super(Conv2DBlock, self).__init__()
        
        # Padding
        if pad_type == 'reflect':
            self.pad = ReflectionPadding2D((padding,padding))
        elif pad_type == 'zero':
            self.pad = tf.keras.layers.ZeroPadding2D(padding)
        
        # Normalization
        self.norm_type = norm
        norm_dim = n_filters
        if norm == 'bn':
            self.norm = tf.keras.layers.BatchNormalization()
        elif norm == 'in':
            self.norm = InstanceNormalization()
            self.norm = None
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2D(norm_dim)
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
    
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self,n_filters, norm='bn', activation='relu', pad_type='zero'):
        super(ResnetIdentityBlock, self).__init__()
        self.norm = norm
        self.model = tf.keras.Sequential()
        for _ in range(2):
            self.model.add(Conv2DBlock(n_filters, 3, 1, 1, pad_type, norm=norm, activation=activation, ))
    def call(self,x):
        res = x
        out = self.model(x)
        out += res
        return out
    
class LinearBlock(tf.keras.Model):
    def __init__(self,out_dim,norm='none',activation='relu'):
        super(LinearBlock,self).__init__()
        use_bias = True
        self.fc = tf.keras.layers.Dense(out_dim,use_bias=use_bias)
        
        if activation == 'relu':
            self.activation = tf.keras.activations.relu
        elif activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh
        else:
            self.activation = None
            
    def call(self,x):
        out = self.fc(x)
        if self.activation:
            out = self.activation(out)
        return out
    
class Conv2D_AdaINBlock(tf.keras.Model):
    def __init__(self, n_filters, ks, st, padding=0, pad_type='zero', use_bias=True, activation='relu', activation_first=False):
        super(Conv2D_AdaINBlock, self).__init__()
        
        # Padding
        if pad_type == 'reflect':
            self.pad = ReflectionPadding2D((padding,padding))
        elif pad_type == 'zero':
            self.pad = tf.keras.layers.ZeroPadding2D(padding)
        
        # Normalization
        self.norm_type = 'adain'
        norm_dim = n_filters
        self.norm = AdaptiveInstanceNorm2D(norm_dim)
        
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
        
    def call(self,x,y):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            x = self.norm(x,y)
        else:
            x = self.conv(self.pad(x))
            x = self.norm(x,y)
            if self.activation:
                x = self.activation(x)
        return x, y
    
class Res_AdaINBlock(tf.keras.Model):
    def __init__(self,n_filters, activation='relu', pad_type='zero'):
        super(Res_AdaINBlock, self).__init__()
        self.norm = 'adain'
        self.model = []
        for _ in range(2):
            self.model.append(Conv2D_AdaINBlock(n_filters, 3, 1, 1, pad_type, activation=activation))
    def call(self,x,y):
        for layer in self.model:
            res = x
            x, _ = layer(x,y)
            x += res
        return x, y