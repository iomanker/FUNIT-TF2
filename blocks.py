import tensorflow as tf
from layers import *

class Conv2DBlock(tf.keras.layers.Layer):
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
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2D(norm_dim)
        else:
            self.norm = None
            
        # Activation
        self.activation_first = activation_first
        if activation == 'relu':
            self.activation = tf.keras.activations.relu
        elif activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh
        else:
            self.activation = None
        
        self.conv = tf.keras.layers.Conv2D(n_filters, ks, st, 'valid', use_bias=use_bias,
                                           kernel_initializer=tf.keras.initializers.he_normal()) #,
                                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        
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

class ResnetIdentityBlock(tf.keras.layers.Layer):
    def __init__(self,n_filters, norm='bn', activation='relu', pad_type='zero'):
        super(ResnetIdentityBlock, self).__init__()
        self.norm = norm
        self.layers = []
        self.layers.append(Conv2DBlock(n_filters, 3, 1, 1, pad_type, norm=norm, activation=activation))
        self.layers.append(Conv2DBlock(n_filters, 3, 1, 1, pad_type, norm=norm, activation='none'))
    def call(self,x):
        res = x
        for l in self.layers:
            x = l(x)
        x += res
        return x

class PreActiResBlock(tf.keras.layers.Layer):
    def __init__(self,in_dim,out_dim,hidden_dim=None,
                 activation='relu',norm='none'):
        super(PreActiResBlock,self).__init__()
        # NEED EXPLANATION for learned_shortcut
        self.learned_shortcut = (in_dim != out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = min(in_dim,out_dim) if hidden_dim is None else hidden_dim
        self.conv_0 = Conv2DBlock(self.hidden_dim,3,1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation='none', activation_first=True)
        self.conv_1 = Conv2DBlock(self.out_dim,3,1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2DBlock(self.out_dim,1,1,
                                      activation='none',use_bias=False)
            
    def call(self,x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        else:
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            x_s = x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = dx + x_s
        return out

class LinearBlock(tf.keras.layers.Layer):
    def __init__(self,out_dim,norm='none',activation='relu'):
        super(LinearBlock,self).__init__()
        use_bias = True
        self.fc = tf.keras.layers.Dense(out_dim,use_bias=use_bias,
                                        kernel_initializer=tf.keras.initializers.he_normal()) # ,
                                        # kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        
        if activation == 'relu':
            self.activation = tf.keras.activations.relu
        elif activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh
        else:
            self.activation = None
            
    def call(self,x):
        out = self.fc(x)
        if self.activation:
            out = self.activation(out)
        return out

class Conv2D_AdaINBlock(tf.keras.layers.Layer):
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
            self.activation = tf.keras.layers.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = tf.keras.activations.tanh
        else:
            self.activation = None
        
        self.conv = tf.keras.layers.Conv2D(n_filters, ks, st, 'valid', use_bias=use_bias,
                                           kernel_initializer=tf.keras.initializers.he_normal()) #,
                                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        
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

class Res_AdaINBlock(tf.keras.layers.Layer):
    def __init__(self,n_filters, activation='relu', pad_type='zero'):
        super(Res_AdaINBlock, self).__init__()
        self.norm = 'adain'
        self.model = [Conv2D_AdaINBlock(n_filters, 3, 1, 1, pad_type, activation=activation),
                      Conv2D_AdaINBlock(n_filters, 3, 1, 1, pad_type, activation='none')]
    def call(self,x,y):
        res = x
        for layer, l_y in zip(self.model, tf.split(y,num_or_size_splits=2,axis=1)):
            x, _ = layer(x,l_y)
        x += res
        return x, y