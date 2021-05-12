import tensorflow as tf

# GAN loss
class GANloss():
    def __init__(self):
        super(GANloss,self).__init__()
    
    @staticmethod
    def gen_loss(resp, feat):
        loss = -tf.reduce_mean(resp, (1,2))
        return loss
    
    # FUNIT/networks.py L71-87
    @staticmethod
    def dis_loss(resp, feat, mode):
        # loss = tf.constant(0.0, tf.float64)
        if mode == 'real':
            loss = tf.keras.activations.relu(1.0 - resp)
            loss = tf.reduce_mean(loss, (1,2))
        elif mode == 'fake':
            loss = tf.keras.activations.relu(1.0 + resp)
            loss = tf.reduce_mean(loss, (1,2))
        return loss
# Reconstruction loss
def recon_loss(a,b):
    loss = tf.reduce_mean(tf.abs(a - b), (1,2,3))
    return loss
# Features matching loss
def featmatch_loss(pred_feat, class_pred_feat):
    pred_feat = tf.reduce_mean(pred_feat,[1,2])
    class_pred_feat = tf.reduce_mean(class_pred_feat,[1,2])
    loss = tf.reduce_mean(tf.abs(pred_feat - class_pred_feat), (1))
    return loss

# Gradient-Penalty Regularization
# https://github.com/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
# def gradient_penalty(d_out, x_in):
#     batch_size = tf.shape(x_in)[0]
#     # https://www.tensorflow.org/api_docs/python/tf/gradients
#     # tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead.
#     grad_dout = tf.gradients(tf.reduce_mean(d_out),x_in)
#     grad_dout2 = grad_dout ** 2
#     assert tf.shape(grad_dout2) == tf.shape(x_in)
#     reg = tf.reduce_sum(grad_dout2) / batch_size
#     return reg

def gradient_penalty(net, x_in, y_in, batch_size):
    # batch_size = tf.shape(x_in)[0]
    batch_size = tf.cast(batch_size, tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_in)
        d_out, _ = net(x_in, y_in)
        pred = tf.reduce_mean(d_out)
    grad_dout = t.gradient(pred, [x_in])[0]
    grad_dout2 = tf.pow(grad_dout, 2)
    # assert tf.shape(grad_dout2) == tf.shape(x_in)
    reg = tf.divide(tf.reduce_sum(grad_dout2), batch_size)
    return reg