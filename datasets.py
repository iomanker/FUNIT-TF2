import tensorflow as tf
import numpy as np
from PIL import Image
import os

def default_filelist_reader(list_filename):
    img_list = []
    with open(list_filename, 'r') as file:
        for line in file.readlines():
            img_path = line.strip()
            img_list.append(img_path)
    return img_list

def default_preprocessing(resize_size, crop_size=(128,128,3), training=True):
    
    if len(crop_size) == 2:
        crop_size = (crop_size[0], crop_size[1], 3)
    if training == False:
        crop_fraction = crop_size[0] / float(resize_size[0])
    # Tensorflow 2.0.0
    # def py_func_read_img(filepath):
    #     filepath = bytes.decode(filepath.numpy())
    #     img = Image.open(filepath).convert('RGB')
    #     img = np.array(img)
    #     return img
    
    def img_transformer(filepath, label):
        raw_img = tf.io.read_file(filepath)
        raw_img = tf.image.decode_jpeg(raw_img,channels=3)
        # raw_img = tf.py_function(func=py_func_read_img, inp=[filepath], Tout=tf.uint8)

        img = tf.image.random_flip_left_right(raw_img)
        img = tf.image.resize(img,resize_size)
        
        if training:
            img = tf.image.random_crop(img, crop_size)
        else:
            img = tf.image.central_crop(img, crop_fraction)
        
        img = tf.cast(img,tf.float32)
        img = (img/255*2) - 1
        return img, label
        
    return img_transformer

# Dataloader
class ImageLabelFilelist(tf.data.Dataset):
    def _generator(img_root, input_list,idx_list):
        for item in zip(input_list,idx_list):
            yield (os.path.join(img_root,item[0]), item[1])
    def __new__(cls, img_root,
                list_filename,
                filelist_reader=default_filelist_reader):
        img_list = filelist_reader(list_filename)
        # NOTICE: You need to check location of class name after split by '/'
        classes = sorted(list(set([path.split('/')[0] for path in img_list])))
        classes_to_idx = {classes[i]: i for i in range(len(classes))}
        # NOTICE: You need to check location of class name after split by '/'
        idx_list = [classes_to_idx[l.split('/')[0]] for l in img_list]
        
        print("Data Loader")
        print("\tRoot: %s" % img_root)
        print("\tList: %s" % list_filename)
        print("\tNumber of classes: %d" % (len(classes)))
        
        return tf.data.Dataset.from_generator(
                cls._generator,
                output_types = (tf.string,tf.uint8),
                args = (img_root,img_list,idx_list,))

def get_tf_dataset(data_folder, data_list,
                   batch_size, crop_size, resize_size, training,
                   num_shuffle, seed=None,
                   preprocessing = default_preprocessing):
    # set datasets we want.
    dataset =  ImageLabelFilelist(data_folder,data_list)
    if num_shuffle > 0:
        dataset = dataset.shuffle(num_shuffle, seed=seed)
    set_img_transformer = default_preprocessing(resize_size, crop_size, training)
        
    return dataset.map(set_img_transformer)

def get_datasets(config, content_seed=None, style_seed=None):
    batch_size = config['batch_size']
    new_size = config['new_size']
    resize_size = (new_size, new_size)
    crop_size = (config['crop_image_height'], config['crop_image_width'], 3)
    num_shuffle = 100000
    train_content_dataset = get_tf_dataset(config['data_folder_train'], config['data_list_train'],
                                           batch_size, crop_size, resize_size, True, num_shuffle, seed=content_seed)
    train_class_dataset = get_tf_dataset(config['data_folder_train'], config['data_list_train'],
                                         batch_size, crop_size, resize_size, True, num_shuffle, seed=style_seed)
    
    test_content_dataset = get_tf_dataset(config['data_folder_test'], config['data_list_test'],
                                          batch_size, crop_size, resize_size, False, num_shuffle, seed=content_seed)
    test_class_dataset = get_tf_dataset(config['data_folder_test'], config['data_list_test'],
                                        batch_size, crop_size, resize_size, False, num_shuffle, seed=style_seed)
    return (train_content_dataset, train_class_dataset,
            test_content_dataset,  test_class_dataset)