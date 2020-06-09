import tensorflow as tf
import argparse
import os
import yaml
import logging
import cv2
from containers import VQVAE_FUNIT
from datasets import get_datasets
from utils import *

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',
                        type=str,
                        default='./configs/funit_animals.yaml',
                        help='configuration file for training and testing')
    parser.add_argument('--ckpt_path', type=str, default='./training_checkpoints')
    parser.add_argument('--output_path', type=str, default='./test_output')
    parser.add_argument('--num_img', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--start', type=int, default=0)
    opts = parser.parse_args()
    
    config = get_config(opts.config)
    
    # Network
    networks = VQVAE_FUNIT(config)
    logging.info("Loaded Network")
    
    # Checkpoint
    checkpoint_dir = opts.ckpt_path
    gen_ckpt_prefix = os.path.join(checkpoint_dir, "gen_ckpt")
    gen_ckpt = tf.train.Checkpoint(optimizer=networks.opt_gen, net=networks.gen)
    gen_ckpt.restore(tf.train.latest_checkpoint(gen_ckpt_prefix))
    logging.info("Loaded ckpt")
    
    # Datasets
    datasets = get_datasets(config)
    test_content_dataset = datasets[2]
    test_class_dataset = datasets[3]
    test_dataset = tf.data.Dataset.zip((test_content_dataset, test_class_dataset))
    get_test_ds = test_dataset.take(opts.num_img).batch(opts.batch_size)
    logging.info("Loaded Datasets")
    
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)
        
    category_img = ['xa','xr','map','xt','xb']
    # for x in category_img:
    #     x_path = os.path.join(opts.output_path, x)
    #     if not os.path.exists(x_path):
    #         os.makedirs(x_path)
    
    test_returns = None
    start = opts.start
    for co_data, cl_data in get_test_ds:
        test_returns = networks.test_step(co_data,cl_data,config)
    
        for idx,x in enumerate(category_img):
            x_path = os.path.join(opts.output_path, x)
            write_images_with_vq((test_returns['xa'],test_returns['xr'],test_returns['map'],test_returns['xt'],test_returns['xb']),
                                 test_returns['display_list'],
                                 os.path.join(opts.output_path, 'test_%02d' % (idx)),
                                 max(config['crop_image_height'], config['crop_image_width']),
                                 config['gen']['vqvae']['num_classes'])
            # write_images_with_vq(images, display_list, filename, square_size=128, classes)
            
            # test_returns[x] = np.uint8(test_returns[x]*127.5+128).clip(0, 255)
            # for idx,img in enumerate(test_returns[x]):
            #     img_path = os.path.join(x_path, '%06d.jpg' % (idx + start))
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     cv2.imwrite(img_path, img)
            #     logging.info("Saved %s" % img_path)
        start += opts.batch_size