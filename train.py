import tensorflow as tf
print("Current Tensorflow Version: %s" % tf.__version__)

import argparse
import time
import sys
import os
from run_step import *
from datasets import *
from containers import *
from losses import *
from utils import *

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
    parser.add_argument('--multigpus',
                        action="store_true")
    parser.add_argument('--test_batch_size',
                         type=int,
                         default=4)
    parser.add_argument('--resume', action="store_true")
    
    opts = parser.parse_args()
    config = get_config(opts.config)
    GLOBAL_BATCH_SIZE = config['batch_size']
    EPOCHS = config['max_iter']
    if opts.batch_size != 0:
        config['batch_size'] = opts.batch_size
    
    # Strategy
    if opts.multigpus:
        print("Multigpus ON.")
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("Multigpus OFF.")
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
    # Datasets
    datasets = get_datasets(config)
    # -- Train
    train_content_dataset = datasets[0]
    train_class_dataset = datasets[1]
    train_dataset = tf.data.Dataset.zip((train_content_dataset, train_class_dataset))
    def train_ds_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(GLOBAL_BATCH_SIZE)
        d = train_dataset.batch(batch_size)
        return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dist_train_dataset = strategy.experimental_distribute_datasets_from_function(train_ds_fn)
    # -- Test
    test_content_dataset = datasets[2]
    test_class_dataset = datasets[3]
    test_dataset = tf.data.Dataset.zip((test_content_dataset, test_class_dataset))
        
    # Tensorboard - Setup
    # import datetime
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = os.path.join(opts.log_path, current_time, 'train')
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
    # Networks & Start Training
    with strategy.scope():
        networks = VQVAE_FUNIT(config)
        test_networks = VQVAE_FUNIT(config)
        
        @tf.function
        def distributed_train_step(dataset_inputs, config):
            dis_per_replica_losses = strategy.experimental_run_v2(networks.dis_train_step, args=(dataset_inputs, config))
            dis_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, dis_per_replica_losses, axis=None)
            
            gen_per_replica_losses = strategy.experimental_run_v2(networks.gen_train_step, args=(dataset_inputs, config))
            gen_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, gen_per_replica_losses, axis=None)
            return gen_loss, dis_loss
        
        # Checkpoint
        checkpoint_dir = opts.ckpt_path
        gen_ckpt_prefix = os.path.join(checkpoint_dir, "gen_ckpt")
        dis_ckpt_prefix = os.path.join(checkpoint_dir, "dis_ckpt")
        gen_ckpt = tf.train.Checkpoint(optimizer= networks.opt_gen, net= networks.gen)
        dis_ckpt = tf.train.Checkpoint(optimizer= networks.opt_dis, net= networks.dis)
        test_gen_ckpt = tf.train.Checkpoint(optimizer= test_networks.opt_gen, net= test_networks.gen)
        
        if opts.resume:
            print("resume ON")
            gen_ckpt.restore(tf.train.latest_checkpoint(gen_ckpt_prefix))
            dis_ckpt.restore(tf.train.latest_checkpoint(dis_ckpt_prefix))
        
        iteration = 0
        for epoch in range(1,EPOCHS+1):
            print("epoch %d: " % epoch)
            try:
                for x in dist_train_dataset:
                    iteration += 1
                    start_time = time.time()
                    G_loss, D_loss = distributed_train_step(x, config)
                    print(" (%d/%d) G_loss: %.4f, D_loss: %.4f, time: %.5f" % (iteration,config['max_iter'],G_loss,D_loss,(time.time() - start_time)))
                    # Tensorboard - Record losses
                    # with train_summary_writer.as_default():
                    #     tf.summary.scalar('g_loss', G_loss, step=iteration)
                    #     tf.summary.scalar('d_loss', D_loss, step=iteration)

                    # Test Step (Print this interval result)
                    if iteration % config['image_save_iter'] == 0 or\
                       iteration % config['image_display_iter'] == 0:
                        gen_ckpt.save(os.path.join(gen_ckpt_prefix, "ckpt"))
                        dis_ckpt.save(os.path.join(dis_ckpt_prefix, "ckpt"))
                        print("load newest ckpt file: %s" % tf.train.latest_checkpoint(gen_ckpt_prefix))
                        test_gen_ckpt.restore(tf.train.latest_checkpoint(gen_ckpt_prefix))
                        if iteration % config['image_save_iter'] == 0:
                            key_str = '%08d' % iteration
                        else:
                            key_str = 'current'
                        output_train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE).take(opts.test_batch_size)
                        output_test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE).take(opts.test_batch_size)
                        for idx,(co_data, cl_data) in output_train_dataset.enumerate():
                            test_returns = test_networks.test_step(co_data,cl_data,config)
                            write_images((test_returns['xa'],test_returns['xr'],test_returns['xt'],test_returns['xb']), 
                                         test_returns['display_list'],
                                         os.path.join(opts.output_path, 'train_%s_%02d' % (key_str, idx)),
                                         max(config['crop_image_height'], config['crop_image_width']))
                        for idx,(co_data, cl_data) in output_test_dataset.enumerate():
                            test_returns = test_networks.test_step(co_data,cl_data,config)
                            write_images((test_returns['xa'],test_returns['xr'],test_returns['xt'],test_returns['xb']), 
                                         test_returns['display_list'],
                                         os.path.join(opts.output_path, 'test_%s_%02d' % (key_str, idx)),
                                         max(config['crop_image_height'], config['crop_image_width']))

                    if iteration >= config['max_iter']:
                        print("End of iteration")
                        break
            except TypeError as err:
                print(err)
                print("Distributed Training doesn't have a functionality of drop_remainder,\n  keep training and still waiting for Tensorflow fixing this problem.")
            if iteration >= config['max_iter']:
                break