import tensorflow as tf
from losses import *
from containers import *
from utils import *

@tf.function
def train_step(nets,co_data,cl_data,config):
    xa, la = co_data
    xb, lb = cl_data
    return_items = {}
    with tf.GradientTape() as d_tape:
        resp_real, real_gen_feat, xt_d, resp_fake, fake_gan_feat = nets.dis_update(co_data,cl_data,config)
        # Discriminator - GAN loss
        l_real = GANloss.dis_loss(resp_real, lb, 'real')
        l_fake = GANloss.dis_loss(resp_fake, lb, 'fake')
        # Discriminator - Gradient Penalty
        l_reg = gradient_penalty(nets.dis, xb, lb)
        
        D_loss = config['gan_w'] * l_real +\
                 config['gan_w'] * l_fake +\
                 10 * l_reg
        
    # Update Gradient
    # - Gradient computing
    dis_grad = d_tape.gradient(D_loss, nets.dis.trainable_variables)
    # - Optimizer
    nets.opt_dis.apply_gradients(zip(dis_grad, nets.dis.trainable_variables))
    
    with tf.GradientTape() as g_tape:
        xt_g, xr, xa_gan_feat, xb_gan_feat = nets.gen_update(co_data,cl_data,config)
        
        resp_xr_fake, xr_gan_feat = nets.dis(xr, la)
        resp_xt_fake, xt_gan_feat = nets.dis(xt_g, lb)
        # Generator - GAN loss
        l_adv_t = GANloss.gen_loss(resp_xt_fake,lb)
        l_adv_r = GANloss.gen_loss(resp_xr_fake,la)
        l_adv = 0.5 * (l_adv_t + l_adv_r)
        # Generator - Reconstruction loss
        l_x_rec = recon_loss(xr, xa)
        # Generator - Feature Matching loss
        l_c_rec = featmatch_loss(xr_gan_feat, xa_gan_feat)
        l_m_rec = featmatch_loss(xt_gan_feat, xb_gan_feat)
        
        G_loss = config['gan_w'] * l_adv +\
                 config['r_w'] * l_x_rec +\
                 config['fm_w'] * (l_c_rec + l_m_rec)
        
    gen_grad = g_tape.gradient(G_loss, nets.gen.trainable_variables)
    nets.opt_gen.apply_gradients(zip(gen_grad, nets.gen.trainable_variables))
    
    return_items['G_loss'] = G_loss
    return_items['D_loss'] = D_loss
    return return_items

def test_step(nets,co_data,cl_data,config):
    xa, la = co_data
    xb, lb = cl_data
    return_items = {}
    xt, xr, xa_gan_feat, xb_gan_feat = nets.gen_update(co_data,cl_data,config)
    return_items['xa'] = xa
    return_items['xb'] = xb
    return_items['xr'] = xr
    return_items['xt'] = xt
    return_items['display_list'] = ['xa','xr','xt','xb']
    return return_items