import tensorflow as tf
# limit GPU growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(len(physical_devices))
assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

import argparse
from run_step import *
from datasets import *
from containers import *
from losses import *
from utils import *

if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config',
#                         type=str,
#                         default='../FUNIT/configs/funit_animals.yaml',
#                         help='configuration file for training and testing')
#     parser.add_argument('--batch_size', type=int, default=0)
#     parser.add_argument('--output_path',
#                         type=str,
#                         default='.',
#                         help="outputs path")
#     parser.add_argument('--test_batch_size',
#                          type=int,
#                          default=4)
#     opts = parser.parse_args()
#     config = get_config(opts.config)
    config = get_config('./configs/funit_animals.yaml')
    
    epochs = config['max_iter']
#     if opts.batch_size != 0:
#         config['batch_size'] = opts.batch_size
    
    # Networks
    networks = FUNIT(config)

    # Datasets
    datasets = get_datasets(config)
    train_content_dataset = datasets[0]
    train_class_dataset = datasets[1]
    train_dataset = tf.data.Dataset.zip((train_content_dataset, train_class_dataset))
    test_content_dataset = datasets[2]
    test_class_dataset = datasets[3]
    test_dataset = tf.data.Dataset.zip((train_content_dataset, train_class_dataset))
    
    # Mean loss
    lossnames = ["G_loss","D_loss"]
    metrics_list = []
    for itemname in lossnames:
        metrics_list.append(tf.keras.metrics.Mean(itemname, dtype=tf.float32))
    
    for epoch in range(epochs):
        print("epoch %d:" % epoch)
        for (co_data, cl_data) in train_dataset:
            train_returns = train_step(networks,co_data,cl_data,config)
            print(" G_loss: %.4f, D_loss: %.4f" % (train_returns['G_loss'],train_returns['D_loss']), end='\r')
            for idx, itemname in enumerate(lossnames):
                metrics_list[idx](train_returns[itemname])
                
        for idx, itemname in enumerate(lossnames):
            print("    {}: {:.4f}".format(itemname,metrics_list[idx].result()))
            metrics_list[idx].reset_states()
            
        if epoch % config['image_save_iter'] == 0 or\
           epoch % config['image_display_iter'] == 0:
            if epoch % config['image_save_iter'] == 0:
                key_str = '%08d' % (epoch + 1)
            else:
                key_str = 'current'
            output_train_dataset = train_dataset.take(opts.test_batch_size)
            output_test_dataset = test_dataset.take(opts.test_batch_size)
            for idx, (co_data, cl_data) in output_train_dataset.enumerate():
                test_returns = test_step(networks,co_data,cl_data,config)
                write_images(zip(test_returns['xa'],test_returns['xr'],test_returns['xt'],test_returns['xb']), 
                             test_returns['display_list'],
                             'train_%s_%02d' % (key_str, idx),
                             max(config['crop_image_height'], config['crop_image_width']))
            for idx, (co_data, cl_data) in output_test_dataset.enumerate():
                test_returns = test_step(networks,co_data,cl_data,config)
                write_images(zip(test_returns['xa'],test_returns['xr'],test_returns['xt'],test_returns['xb']), 
                             test_returns['display_list'],
                             'test_%s_%02d' % (key_str, idx),
                             max(config['crop_image_height'], config['crop_image_width']))