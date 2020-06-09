import tensorflow as tf
from blocks import *
from layers import *
from models import *
from losses import *
import numpy as np
import sys

class VQVAE_Generator(tf.keras.Model):
    def __init__(self, config):
        super(VQVAE_Generator, self).__init__()
        n_features     = config['nf']
        n_features_mlp = config['nf_mlp']
        down_class     = config['n_downs_class']
        down_content   = config['n_downs_content']
        n_mlp_blocks   = config['n_mlp_blks']
        n_res_blocks   = config['n_res_blks']
        latent_dim     = config['latent_dim']
        
        self.E_content = ContentEncoder(downs=down_content,
                                        n_res=n_res_blocks,
                                        n_filters=n_features,
                                        norm='in',activation='relu',pad_type='reflect')
        # VQVAE for Content Encoder
        
        # D - This value is not that important, usually 64 works.
        #   - This will not change the capacity in the information-bottleneck.
        self.embedding_dim = config['vqvae']['dim_class']
        # K - The higher this value, the higher the capacity in the information bottleneck.
        self.num_embeddings = config['vqvae']['num_classes']
        self.commitment_cost = config['vqvae']['commitment_cost']
        self.pre_vqvae = tf.keras.layers.Conv2D(self.embedding_dim, kernel_size=1, strides=1, padding='valid')
        self.vq = VectorQuantizer(self.embedding_dim, self.num_embeddings, self.commitment_cost)
        
        self.E_class = ClassEncoder(downs=down_class,
                                    latent_dim=self.embedding_dim,
                                    n_filters=n_features,
                                    norm='none',activation='relu',pad_type='reflect')
        
        # self.E_content.output_filters*2
        # self.mlp = MLP(out_dim=self.embedding_dim*2,
        #                dim=n_features_mlp,
        #                n_blk=n_mlp_blocks, activation='relu')
        
        self.Dec = Decoder(ups=down_content,
                           n_res=n_res_blocks,
                           n_filters=self.embedding_dim,
                           out_dim=3,
                           activation='relu',pad_type='reflect')
        
    def decode(self, content, model_code):
        # adain_params = self.mlp(model_code)
        adain_params = model_code
        imgs = self.Dec(content,adain_params)
        return imgs
    
class VQVAE_FUNIT(tf.keras.Model):
    def __init__(self, config):
        super(VQVAE_FUNIT, self).__init__()
        self.gen = VQVAE_Generator(config['gen'])
        self.dis = Discriminator(config['dis'])
        self.opt_gen = tf.keras.optimizers.RMSprop(learning_rate=config['lr_gen'])
        self.opt_dis = tf.keras.optimizers.RMSprop(learning_rate=config['lr_dis'])
        
    def gen_train_step(self, x, config):
        co_data, cl_data = x
        xa, la = co_data
        xb, lb = cl_data
        with tf.GradientTape() as g_tape, tf.GradientTape() as vq_tape:
            xt_g, xr, xa_gan_feat, xb_gan_feat, vqvae_loss, _ = self.gen_produce(co_data, cl_data, config, True)
            
            resp_xr_fake, xr_gan_feat = self.dis(xr, la)
            resp_xt_fake, xt_gan_feat = self.dis(xt_g, lb)
            
            # GAN loss
            l_adv_t = GANloss.gen_loss(resp_xt_fake,lb)
            l_adv_r = GANloss.gen_loss(resp_xr_fake,la)
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            # Reconstruction loss
            l_x_rec = recon_loss(xr, xa)
            l_x_rec = tf.reduce_mean(l_x_rec)
            # Feature Matching loss
            l_c_rec = featmatch_loss(xr_gan_feat, xa_gan_feat)
            l_c_rec = tf.reduce_mean(l_c_rec)
            l_m_rec = featmatch_loss(xt_gan_feat, xb_gan_feat)
            l_m_rec = tf.reduce_mean(l_m_rec)
            l_fm_rec = l_c_rec + l_m_rec
            
            G_loss = config['gan_w'] * l_adv + \
                     config['r_w'] * l_x_rec + \
                     config['fm_w'] * l_fm_rec + \
                     config['vq_w'] * vqvae_loss
            # G_loss = config['r_w'] * l_x_rec + config['vq_w'] * vqvae_loss
            # VQ_loss = config['vq_w'] * vqvae_loss
            '''tf.print('gan: ', config['gan_w'] * l_adv,
                     ', recon: ', config['r_w'] * l_x_rec,
                     ', fm: ', config['fm_w'] * l_fm_rec,
                     ', vq: ', config['vq_w'] * vqvae_loss,
                     output_stream=sys.stdout)'''
            loss = G_loss * (1.0 / config['batch_size'])
        # ###
        gen_grad = g_tape.gradient(loss, self.gen.trainable_variables)
        # ###
        self.opt_gen.apply_gradients(zip(gen_grad, self.gen.trainable_variables))
        return G_loss
            
    def gen_produce(self, co_data, cl_data, config, training):
        xa,la = co_data
        xb,lb = cl_data
        class_xa = self.gen.E_content(xa)
        # [16, 16, 16, 512]
        # tf.print(tf.shape(class_xa))
        class_pre_vqresult = self.gen.pre_vqvae(class_xa)
        class_vq = self.gen.vq(class_pre_vqresult, training)
        
        style_xa = self.gen.E_class(xa)
        style_xb = self.gen.E_class(xb)
        
        # [16 8 8 256]
        # tf.print(tf.shape(style_xb))
        xt = self.gen.decode(class_vq['quantize'], style_xb) # Translation
        xr = self.gen.decode(class_vq['quantize'], style_xa) # Reconstruction
        
        _, xa_gan_feat = self.dis(xa,la)
        _, xb_gan_feat = self.dis(xb,lb)
        return xt, xr, xa_gan_feat, xb_gan_feat, class_vq['loss'], class_vq['encoding_indices']
    
    def dis_train_step(self, x, config):
        co_data, cl_data = x
        xa, la = co_data
        xb, lb = cl_data
        with tf.GradientTape() as d_tape:
            resp_real, real_gen_feat, xt_d, resp_fake, fake_gan_feat =\
                                                self.dis_produce(co_data,cl_data,config,True)
            # Discriminator - GAN loss
            l_real = GANloss.dis_loss(resp_real, lb, 'real')
            l_fake = GANloss.dis_loss(resp_fake, lb, 'fake')
            # Discriminator - Gradient Penalty
            l_reg = gradient_penalty(self.dis, xb, lb)

            D_loss = config['gan_w'] * l_real +\
                     config['gan_w'] * l_fake +\
                     10 * l_reg
            loss = D_loss * (1.0 / config['batch_size'])
        dis_grad = d_tape.gradient(loss, self.dis.trainable_variables)
        self.opt_dis.apply_gradients(zip(dis_grad, self.dis.trainable_variables))
        return D_loss
    
    def dis_produce(self, co_data, cl_data, config, training):
        xa,la = co_data
        xb,lb = cl_data
        resp_real, real_gan_feat = self.dis(xb,lb)
        class_xa = self.gen.E_content(xa)
        class_pre_vqresult = self.gen.pre_vqvae(class_xa)
        class_vq = self.gen.vq(class_pre_vqresult, training)
        
        style_xb = self.gen.E_class(xb)
        xt = self.gen.decode(class_vq['quantize'], style_xb)
        resp_fake, fake_gan_feat = self.dis(xt,lb)
        return resp_real, real_gan_feat, xt, resp_fake, fake_gan_feat
    
    def test_step(self, co_data, cl_data, config):
        xa, la = co_data
        xb, lb = cl_data
        return_items = {}
        xt, xr, xa_gan_feat, xb_gan_feat, _, encoding_indices = self.gen_produce(co_data,cl_data,config,False)
        return_items['xa'] = xa.numpy()
        return_items['xb'] = xb.numpy()
        return_items['xr'] = xr.numpy()
        return_items['xt'] = xt.numpy()
        return_items['map'] = encoding_indices.numpy()
        return_items['display_list'] = ['xa','xr','map','xt','xb']
        return return_items

class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator,self).__init__()
        n_features     = config['nf']
        n_features_mlp = config['nf_mlp']
        down_class     = config['n_downs_class']
        down_content   = config['n_downs_content']
        n_mlp_blocks   = config['n_mlp_blks']
        n_res_blocks   = config['n_res_blks']
        latent_dim     = config['latent_dim']
        # Content Encoder(Ex)
        self.E_content = ContentEncoder(downs=down_content,
                                        n_res=n_res_blocks,
                                        n_filters=n_features,
                                        norm='in',activation='relu',pad_type='reflect')
        # Class Encoder(Ey->Zy)
        self.E_class = ClassEncoder(downs=down_class,
                                    latent_dim=latent_dim,
                                    n_filters=n_features,
                                    norm='none',activation='relu',pad_type='reflect')
        # Adaptively computing using Zy
        self.mlp = MLP(out_dim=self.E_content.output_filters*2,
                       dim=n_features_mlp,
                       n_blk=n_mlp_blocks, activation='relu')
        # Decoder
        self.Dec = Decoder(ups=down_content,
                           n_res=n_res_blocks,
                           n_filters=self.E_content.output_filters,
                           out_dim=3,
                           activation='relu',pad_type='reflect')
    def decode(self, content, model_code):
        adain_params = self.mlp(model_code)
        imgs = self.Dec(content,adain_params)
        return imgs
    
class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator,self).__init__()
        assert config['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = config['n_res_blks'] // 2
        nf = config['nf']
        
        self.model = tf.keras.Sequential()
        self.model.add(Conv2DBlock(nf,7,1,
                                   3,pad_type='reflect',
                                   norm='none',activation='none'))
        for i in range(self.n_layers - 1):
            nf_out = min(nf * 2, 1024)
            self.model.add(PreActiResBlock(nf,nf,None,'leakyrelu','none'))
            self.model.add(PreActiResBlock(nf,nf_out,None,'leakyrelu','none'))
            self.model.add(ReflectionPadding2D((1,1)))
            self.model.add(tf.keras.layers.AveragePooling2D(pool_size=3,strides=2))
            nf = min(nf * 2, 1024)
        nf_out = min(nf * 2, 1024)
        self.model.add(PreActiResBlock(nf,nf,None,'leakyrelu','none'))
        self.model.add(PreActiResBlock(nf,nf_out,None,'leakyrelu','none'))
        self.last_layer = Conv2DBlock(n_filters=config['num_classes'], 
                                      ks=1, st=1,
                                      norm="none", activation='leakyrelu', activation_first=True)
        
    def call(self,x,y):
        # assert tf.shape(x)[0] == tf.shape(y)[0]
        feat = self.model(x)
        out = self.last_layer(feat)
        y_idx = tf.cast(tf.range(0,tf.shape(y)[0]), tf.int32)
        y_idx = tf.stack([y_idx, tf.cast(y, tf.int32)], axis=1)
        # y_idx = np.array(range(0,tf.shape(y)[0]))
        # y_idx = np.array(list(zip(y_idx,y.numpy())))
        out = tf.transpose(out, perm=[0, 3, 1, 2]) 
        out = tf.gather_nd(out, y_idx)
        return out, feat
    
class FUNIT(tf.keras.Model):
    def __init__(self, config):
        super(FUNIT,self).__init__()
        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])
        self.opt_gen = tf.keras.optimizers.RMSprop(learning_rate=config['lr_gen'])
        self.opt_dis = tf.keras.optimizers.RMSprop(learning_rate=config['lr_dis'])
        
    def gen_update(self, co_data, cl_data, config):
        xa,la = co_data
        xb,lb = cl_data
        class_xa = self.gen.E_content(xa)
        style_xa = self.gen.E_class(xa)
        style_xb = self.gen.E_class(xb)
        
        xt = self.gen.decode(class_xa, style_xb) # Translation
        xr = self.gen.decode(class_xa, style_xa) # Reconstruction
        
        _, xa_gan_feat = self.dis(xa,la)
        _, xb_gan_feat = self.dis(xb,lb)
        return xt, xr, xa_gan_feat, xb_gan_feat
    
    def dis_update(self, co_data, cl_data, config):
        xa,la = co_data
        xb,lb = cl_data
        resp_real, real_gan_feat = self.dis(xb,lb)
        class_xa = self.gen.E_content(xa)
        style_xb = self.gen.E_class(xb)
        xt = self.gen.decode(class_xa, style_xb)
        resp_fake, fake_gan_feat = self.dis(xt,lb)
        return resp_real, real_gan_feat, xt, resp_fake, fake_gan_feat