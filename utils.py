import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import yaml

def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy())
    plt.axis('off')
    
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
    
def write_images(images, display_list, filename, square_size=128):
    # suppose images shape is zip(xa,xr,xt,xb) = [(xa[0],xr[0],xt[0],xb[0]),...]
    # display_list = ['xa','xr','xt','xb']
    category_imgs = len(images) # 4
    batch_size = len(images[0])
    nrow = category_imgs
    ncol = batch_size
    fig = plt.figure(figsize=(ncol+1, nrow+1), dpi=square_size)
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.0, hspace=0.0, 
             top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
             left=0.5/(ncol+1), right=1-0.5/(ncol+1))
    for j in range(ncol):
        for i in range(nrow):
            ax = plt.subplot(gs[i,j])
            ax.imshow(images[i][j])
            # convert to [0..1]
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis=u'both', which=u'both',length=0)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0)
            if j == 0:
                plt.ylabel(display_list[i])
    plt.savefig(filename + '.png',bbox_inches='tight')