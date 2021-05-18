import tensorflow as tf
from blocks import *
from layers import *
from models import *
from losses import *
import numpy as np

class Generator(tf.keras.layers.Layer):
    def __init__(self, nfs, downs, blocks, latent_dim, **kwargs):
        super(Generator,self).__init__(**kwargs)
        n_features,   n_features_mlp = nfs
        down_class,   down_content   = downs
        n_mlp_blocks, n_res_blocks   = blocks
        
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
        self.mlp = MLP(out_dim=self.E_content.output_filters*2*(n_res_blocks*2),
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
    
    def call(self, co_data, cl_data):
        xa, la = co_data
        xb, lb = cl_data
        content_xa = self.E_content(xa)
        
        style_xa = self.E_class(xa)
        style_xb = self.E_class(xb)
        
        xt = self.decode(content_xa, style_xb) # Translation
        xr = self.decode(content_xa, style_xa) # Reconstruction
        
        return xt, xr
    
class Discriminator(tf.keras.layers.Layer):
    def __init__(self, n_features, n_res_blocks, n_classes):
        super(Discriminator,self).__init__()
        assert n_res_blocks % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = n_res_blocks // 2
        nf = n_features
        
        self.layers = []
        self.layers.append(Conv2DBlock(nf,7,1,
                                   3,pad_type='reflect',
                                   norm='none',activation='none'))
        for i in range(self.n_layers - 1):
            nf_out = min(nf * 2, 1024)
            self.layers.append(PreActiResBlock(nf,nf,None,'leakyrelu','none'))
            self.layers.append(PreActiResBlock(nf,nf_out,None,'leakyrelu','none'))
            self.layers.append(ReflectionPadding2D((1,1)))
            # self.layers.append(tf.keras.layers.ZeroPadding2D((1,1)))
            self.layers.append(tf.keras.layers.AveragePooling2D(pool_size=3,strides=2))
            nf = min(nf * 2, 1024)
        nf_out = min(nf * 2, 1024)
        self.layers.append(PreActiResBlock(nf,nf,None,'leakyrelu','none'))
        self.layers.append(PreActiResBlock(nf,nf_out,None,'leakyrelu','none'))
        self.last_layer = Conv2DBlock(n_filters=n_classes, 
                                      ks=1, st=1,
                                      norm="none", activation='none', activation_first=True)
        
    def call(self,x,y):
        # assert tf.shape(x)[0] == tf.shape(y)[0]
        feat = x
        for l in self.layers:
            feat = l(feat)
        feat = tf.keras.layers.LeakyReLU(0.2)(feat)
        out = self.last_layer(feat)
        y_idx = tf.cast(tf.range(0,tf.shape(y)[0]), tf.int32)
        y_idx = tf.stack([y_idx, tf.cast(y, tf.int32)], axis=1)
        # y_idx = np.array(range(0,tf.shape(y)[0]))
        # y_idx = np.array(list(zip(y_idx,y.numpy())))
        out = tf.transpose(out, perm=[0, 3, 1, 2]) 
        out = tf.gather_nd(out, y_idx)
        # feat shape: [B, 8, 8, 1024]
        return out, feat
    
class FUNIT(tf.keras.Model):
    def __init__(self, config):
        super(FUNIT,self).__init__()
        
        # all arguments
        gen_nfs     = (config['gen']['nf'], config['gen']['nf_mlp'])
        gen_downs   = (config['gen']['n_downs_class'], config['gen']['n_downs_content'])
        gen_blocks  = (config['gen']['n_mlp_blks'], config['gen']['n_res_blks'])
        gen_latent_dim     = config['gen']['latent_dim']
        
        dis_n_features     = config['dis']['nf']
        dis_n_res_blocks   = config['dis']['n_res_blks']
        dis_n_classes      = config['dis']['num_classes']
        
        self.dict_w = {'gan': config['gan_w'], 'rec': config['r_w'], 'fm': config['fm_w']}
        
        self.gen = Generator(gen_nfs, gen_downs, gen_blocks, gen_latent_dim)
        self.gen_test = Generator(gen_nfs, gen_downs, gen_blocks, gen_latent_dim)
        self.dis = Discriminator(dis_n_features, dis_n_res_blocks, dis_n_classes)
        self.opt_gen = tf.keras.optimizers.RMSprop(learning_rate=config['lr_gen'], rho=0.99, epsilon=1e-8)
        self.opt_dis = tf.keras.optimizers.RMSprop(learning_rate=config['lr_dis'], rho=0.99, epsilon=1e-8)
        
        self.GLOBAL_BATCH_SIZE = config['batch_size']
        self.weight_decay = config['weight_decay']
        self.build((config['crop_image_height'],config['crop_image_width']))
        
    def build(self, size=(128,128)):
        co_data = (tf.random.normal((1,size[0],size[1],3)), tf.convert_to_tensor(np.array([0])))
        cl_data = (tf.random.normal((1,size[0],size[1],3)), tf.convert_to_tensor(np.array([0])))
        self.gen(co_data, cl_data)
        self.gen_test(co_data, cl_data)
        self.gen_test.set_weights(self.gen.get_weights())
        
    @tf.function
    def distributed_train_step(self, dataset_inputs, strategy):
        co_data, cl_data = dataset_inputs
        
        # Tensorflow 2.4.0
        per_replica_G_losses, per_replica_D_losses, \
        = strategy.run(self.train_step, args=(co_data, cl_data))
        G_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_G_losses, axis=None)
        D_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_D_losses, axis=None)
        return G_loss, D_loss
    
    def train_step(self, co_data, cl_data):
        D_loss, _ = self.dis_update(co_data, cl_data)
        G_loss, _ = self.gen_update(co_data, cl_data)
        return G_loss, D_loss
    
    def gen_update(self, co_data, cl_data):
        xa, la = co_data
        xb, lb = cl_data
        with tf.GradientTape() as g_tape:
            xt, xr = self.gen(co_data, cl_data)
            
            _, xa_gan_feat = self.dis(xa,la)
            _, xb_gan_feat = self.dis(xb,lb)
            resp_xr_fake, xr_gan_feat = self.dis(xr,la)
            resp_xt_fake, xt_gan_feat = self.dis(xt,lb)
            
            # Generator losses
            l_adv_t = GANloss.gen_loss(resp_xt_fake, lb)
            l_adv_r = GANloss.gen_loss(resp_xr_fake, la)
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            l_adv = tf.nn.compute_average_loss(l_adv, global_batch_size=self.GLOBAL_BATCH_SIZE)
            
            l_x_rec = recon_loss(xr, xa)
            l_x_rec = tf.nn.compute_average_loss(l_x_rec, global_batch_size=self.GLOBAL_BATCH_SIZE)
            
            l_c_rec = featmatch_loss(xr_gan_feat, xa_gan_feat)
            l_m_rec = featmatch_loss(xt_gan_feat, xb_gan_feat)
            l_fm_rec = l_c_rec + l_m_rec
            l_fm_rec = tf.nn.compute_average_loss(l_fm_rec, global_batch_size=self.GLOBAL_BATCH_SIZE)
            
            G_loss = self.dict_w['gan'] * l_adv + self.dict_w['rec'] * l_x_rec + self.dict_w['fm'] * l_fm_rec
            # print("G_loss: ", G_loss.numpy())
            # print("  GAN: ", (self.dict_w['gan'] * l_adv).numpy(),
            #       "Rec: ", (self.dict_w['rec'] * l_x_rec).numpy(),
            #       "Feat xr: ", (self.dict_w['fm'] * l_c_rec).numpy(),
            #       "Feat xt: ", (self.dict_w['fm'] * l_m_rec).numpy())
        gen_grad = g_tape.gradient(G_loss, self.gen.trainable_variables)
        # Weight Decay
        for i in range(len(gen_grad)):
            gen_grad[i] = gen_grad[i] + (self.weight_decay * self.gen.trainable_variables[i])
        self.opt_gen.apply_gradients(zip(gen_grad, self.gen.trainable_variables))
        return G_loss, gen_grad
        
    def dis_update(self, co_data, cl_data):
        xa, la = co_data
        xb, lb = cl_data
        with tf.GradientTape() as d_tape:
            xt, _ = self.gen(co_data, cl_data)
            resp_real, _ = self.dis(xb,lb)
            resp_xt_fake, _ = self.dis(tf.stop_gradient(xt),lb)
            
            # Discriminator losses
            l_real = GANloss.dis_loss(resp_real, lb, 'real')
            l_fake = GANloss.dis_loss(resp_xt_fake, lb, 'fake')
            l_reg = gradient_penalty(self.dis, xb, lb, self.GLOBAL_BATCH_SIZE)
            
            l_real = self.dict_w['gan'] * l_real
            l_real = tf.nn.compute_average_loss(l_real, global_batch_size=self.GLOBAL_BATCH_SIZE)
            l_fake = self.dict_w['gan'] * l_fake
            l_fake = tf.nn.compute_average_loss(l_fake, global_batch_size=self.GLOBAL_BATCH_SIZE)
            l_reg = 10 * l_reg
            # l_reg = tf.nn.compute_average_loss(l_reg, global_batch_size=self.GLOBAL_BATCH_SIZE)
            
            D_loss = l_real + l_reg + l_fake
            # print("D_loss: ", D_loss)
            # tf.print('  Real:', l_real.numpy(), ' Fake:', l_fake.numpy(), 'Penalty:', l_reg.numpy())
        dis_grad = d_tape.gradient(D_loss, self.dis.trainable_variables)
        # Weight Decay
        for i in range(len(dis_grad)):
            dis_grad[i] = dis_grad[i] + (self.weight_decay * self.dis.trainable_variables[i])
        self.opt_dis.apply_gradients(zip(dis_grad, self.dis.trainable_variables))
        return D_loss, dis_grad
    
    def test_step(self, co_data, cl_data):
        xa, la = co_data
        xb, lb = cl_data
        return_items = {}
        xt, xr = self.gen(co_data, cl_data)
        return_items['xa'] = xa.numpy()
        return_items['xb'] = xb.numpy()
        return_items['xr'] = xr.numpy()
        return_items['xt'] = xt.numpy()
        return return_items
    
    def test_EMA_step(self, co_data, cl_data):
        xa, la = co_data
        xb, lb = cl_data
        return_items = {}
        xt, xr = self.gen_test(co_data, cl_data)
        return_items['xa'] = xa.numpy()
        return_items['xb'] = xb.numpy()
        return_items['xr'] = xr.numpy()
        return_items['xt'] = xt.numpy()
        return return_items
    
    def call(self, co_data, cl_data):
        xa, la = co_data
        xb, lb = cl_data
        _, _ = self.gen(co_data, cl_data)
        _, _ = self.dis(xa, la)
    
    def compute_style_code(self, style_batch):
        # style_batch = [1, 128, 128, 3]
        code_style_batch = self.gen_test.E_class(style_batch)
        return code_style_batch
    
    def translate_simple(self, content_image, class_code):
        # content_image = [1, 128, 128, 3]
        # class_code = [1, 64]
        content_fea = self.gen_test.E_content(content_image)
        xt = self.gen_test.decode(content_fea, class_code)
        return xt