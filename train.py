import tensorflow as tf
print("Current Tensorflow Version: %s" % tf.__version__)

import argparse
import time
import datetime
import sys
import os
from datasets import *
from containers import *
from losses import *
from utils import *

def set_tensorboard(log_path):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_path, current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    return train_summary_writer

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='./configs/funit_animals.yaml',
                        help='configuration file for training and testing')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--output_path',
                        type=str,
                        default='./outputs',
                        help="outputs path")
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='./training_checkpoints',
                        help="checkpoint path")
    parser.add_argument('--log_path', type=str,
                        default="../tensorflow/logs/")
    parser.add_argument('--multigpus',
                        action="store_true")
    parser.add_argument('--test_batch_size',
                         type=int,
                         default=4)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--start_iter', type=int, default=1)
    
    opts = parser.parse_args()
    CONFIG = get_config(opts.config)
    BATCH_SIZE_PER_REPLICA = CONFIG['batch_size']
    MAX_ITER = CONFIG['max_iter']
    DISPLAY_MAX = max(CONFIG['crop_image_height'], CONFIG['crop_image_width'])
    if opts.batch_size != 0:
        CONFIG['batch_size'] = opts.batch_size
    
    datasets = get_datasets(CONFIG)
    train_dataset = tf.data.Dataset.zip((datasets[0], datasets[1]))
    test_dataset = tf.data.Dataset.zip((datasets[2], datasets[3]))
    
    # checkpoint
    checkpoint_dir = opts.ckpt_path
    gen_ckpt_prefix = os.path.join(checkpoint_dir, "gen_ckpt")
    dis_ckpt_prefix = os.path.join(checkpoint_dir, "dis_ckpt")
    
    
    if opts.multigpus:
        tf.print("Multigpus ON.")
        strategy = tf.distribute.MirroredStrategy()
    else:
        tf.print("Multigpus OFF.")
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        network = FUNIT(CONFIG)
        network.GLOBAL_BATCH_SIZE = CONFIG['batch_size'] * strategy.num_replicas_in_sync
        
        # checkpoint
        gen_ckpt = tf.train.Checkpoint(optimizer= network.opt_gen, net= network.gen)
        dis_ckpt = tf.train.Checkpoint(optimizer= network.opt_dis, net= network.dis)
        
    gen_ckpt_manager = tf.train.CheckpointManager(gen_ckpt, gen_ckpt_prefix, max_to_keep=2)
    dis_ckpt_manager = tf.train.CheckpointManager(dis_ckpt, dis_ckpt_prefix, max_to_keep=2)
        
    if opts.resume:
        print("resume ON")
        gen_ckpt.restore(gen_ckpt_manager.latest_checkpoint)
        dis_ckpt.restore(dis_ckpt_manager.latest_checkpoint)
    
    def train_ds_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(network.GLOBAL_BATCH_SIZE)
        d = train_dataset.batch(batch_size)
        return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    train_dist_dataset = strategy.experimental_distribute_datasets_from_function(train_ds_fn)
    display_train_dataset = train_dataset.batch(4).take(opts.test_batch_size)
    test_dataset = test_dataset.batch(4).take(opts.test_batch_size)
        
    train_summary_writer = set_tensorboard(opts.log_path)
    
    train_iter = iter(train_dist_dataset)
    GLOBAL_BATCH_SIZE = network.GLOBAL_BATCH_SIZE
    for num_iter in range(opts.start_iter,MAX_ITER+1):
        start_time = time.time()
        G_loss, D_loss = network.distributed_train_step(next(train_iter), strategy)
        print(" (%d/%d) G_loss: %.4f, D_loss: %.4f, time: %.5f" % (num_iter,MAX_ITER,G_loss,D_loss,(time.time() - start_time)))
        
        # Output intermediate image results
        if num_iter % CONFIG['image_save_iter'] == 0 or \
           num_iter % CONFIG['image_display_iter'] == 0:
            if num_iter % CONFIG['image_save_iter'] == 0:
                key_str = '%08d' % num_iter
            else:
                key_str = 'current'
            save_result(network, display_train_dataset, key_str, opts.output_path, DISPLAY_MAX, True)
            save_result(network, test_dataset, key_str, opts.output_path, DISPLAY_MAX, False)
        
        # checkpoint snapshot save
        if num_iter % CONFIG['snapshot_save_iter'] == 0:
            gen_path = gen_ckpt_manager.save()
            print("save generator checkpoint: ", gen_path)
            dis_path = dis_ckpt_manager.save()
            print("save discriminator checkpoint: ", dis_path)