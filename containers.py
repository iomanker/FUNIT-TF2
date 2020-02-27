import tensorflow as tf
from blocks import *
from layers import *
from models import *
class Generator(tf.keras.Model):
    def __init__(self, config):
        n_features     = config['nf']
        n_features_mlp = config['nf_mlp']
        down_class     = config['n_down_class']
        down_content   = config['n_downs_content']
        n_mlp_blocks   = config['n_mlp_blks']
        n_res_blocks   = config['n_res_blks']
        latent_dim     = config['latent_dim']
        # Content Encoder(Ex)
        self.E_content = ContentEncoder(downs=down_class,
                                        n_res=n_res_blocks,
                                        n_filters=n_features,
                                        norm='in',activation='relu',pad_type='reflect')
        # Class Encoder(Ey->Zy)
        self.E_class = ClassEncoder(downs=down_content,
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
        imgs = self.Dec(content,model_code)
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
            nf_out = np.min([nf * 2, 1024])
            self.model.add(PreActiResBlock(nf,nf,None,'leakyrelu','none'))
            self.model.add(PreActiResBlock(nf,nf_out,None,'leakyrelu','none'))
            self.model.add(ReflectionPadding2D((1,1)))
            self.model.add(tf.keras.layers.AveragePooling2D(pool_size=3,strides=2))
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        self.model.add(PreActiResBlock(nf,nf,None,'leakyrelu','none'))
        self.model.add(PreActiResBlock(nf,nf_out,None,'leakyrelu','none'))
        self.last_layer = Conv2DBlock(n_filters=config['num_classes'], 
                                      ks=1, st=1,
                                      norm="none", activation='leakyrelu', activation_first=True)
        
    def call(self,x,y):
        assert tf.shape(x)[0] == tf.shape(y)[0]
        feat = self.model(x)
        out = self.last_layer(feat)
        idx = tf.convert_to_tensor(range(tf.shape(out)[0]), dtype=tf.float32)
        out = out[idx, :, :, y]
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
        xr = self.gen.deocde(class_xa, style_xa) # Reconstruction
        
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