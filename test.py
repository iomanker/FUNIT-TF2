import tensorflow as tf
import argparse
import os
import yaml
import logging
from containers import FUNIT
from datasets import get_datasets
from utils import *

def save_result(network, dataset, key_str, save_root, display_max, isTrain = False):
    filename = "train" if isTrain else "test"
    filename = filename + "_%s_%02d"
    for idx, (co_data, cl_data) in dataset.enumerate():
        return_dict = network.test_step(co_data, cl_data)
        display_list = []
        display_imgs = []
        for key in return_dict:
            display_list.append(key)
            display_imgs.append(return_dict[key])
        write_images(display_imgs, display_list,
                     os.path.join(save_root, filename % (key_str, idx)),
                     display_max)

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
    DISPLAY_MAX = max(config['crop_image_height'], config['crop_image_width'])
    
    # Network
    network = FUNIT(config)
    logging.info("Loaded Network")
    
    # Checkpoint
    checkpoint_dir = opts.ckpt_path
    gen_ckpt_prefix = os.path.join(checkpoint_dir, "gen_ckpt")
    gen_ckpt = tf.train.Checkpoint(optimizer=network.opt_gen, net=network.gen)
    gen_ckpt.restore(tf.train.latest_checkpoint(gen_ckpt_prefix))
    logging.info("Loaded ckpt")
    
    # Datasets
    datasets = get_datasets(config, content_seed=1234, style_seed=5678)
    train_dataset = tf.data.Dataset.zip((datasets[0], datasets[1]))
    test_dataset = tf.data.Dataset.zip((datasets[2], datasets[3]))
    train_dataset = train_dataset.take(opts.num_img).batch(opts.batch_size)
    test_dataset = test_dataset.take(opts.num_img).batch(opts.batch_size)
    logging.info("Loaded Datasets")
    
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)
    save_result(network, train_dataset, "test", opts.output_path, DISPLAY_MAX, True)
    save_result(network, test_dataset, "test", opts.output_path, DISPLAY_MAX, False)