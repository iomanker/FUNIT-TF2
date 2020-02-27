import tensorflow as tf

# GAN loss
class GANloss():
    def __init__(self):
        super(GANloss,self).__init__()
    
    @staticmethod
    def gen_loss(self, resp, feat):
        loss = tf.negative(tf.reduce_mean(resp))
        return loss
    
    # FUNIT/networks.py L71-87
    @staticmethod
    def dis_loss(self, resp, feat, mode):
        loss = tf.constant(0.0, tf.float64)
        if mode == 'real':
            loss = tf.keras.activations.relu(1.0 - resp)
            loss = tf.reduce_mean(loss)
        elif mode == 'fake':
            loss = tf.keras.activations.relu(1.0 + resp)
            loss = tf.reduce_mean(loss)
        return loss
# Reconstruction loss
def recon_loss(a,b):
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(a,b)
# Features matching loss
def featmatch_loss(pred_feat, class_pred_feat):
    pred_feat = tf.reduce_mean(pred_feat,2)
    pred_feat = tf.reduce_mean(pred_feat,1)
    class_pred_feat = tf.reduce_mean(class_pred_feat,2)
    class_pred_feat = tf.reduce_mean(class_pred_feat,1)
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(class_pred_feat,pred_feat)

# Gradient-Penalty Regularization
# https://github.com/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
def gradient_penalty(d_out, x_in):
    batch_size = tf.shape(x_in)[0]
    grad_dout = tf.gradients(tf.reduce_mean(d_out),x_in)
    grad_dout2 = grad_dout ** 2
    assert tf.shape(grad_dout2) == tf.shape(x_in)
    reg = tf.reduce_sum(grad_dout2) / batch_size
    return reg