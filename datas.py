import tensorflow as tf

def default_filelist_reader(list_filename):
    img_list = []
    with open(list_filename, 'r') as file:
        for line in file.readlines():
            img_path = line.strip()
            img_list.append(img_path)
    return img_list

# torchvision.transform ToTensor & Normalize
# https://netoken27.blogspot.com/2019/03/pytorch-torchvisiontransforms.html
def default_img_processor(filepath,label):
    raw_img = tf.io.read_file(filepath)
    raw_img = tf.image.decode_jpeg(raw_img,channels=3)
    
    # [0,1]
    img = tf.image.convert_image_dtype(raw_img,tf.float32)
    # [-1,1]
    img = tf.image.per_image_standardization(img)
    return img, label

def default_img_transformer(img,crop_size,resize_size):
    # crop
    img_shape = tf.shape(img)
    img = tf.image.random_crop(img,size=[tf.minimum(crop_size[0],img_shape[0]),tf.minimum(crop_size[1],img_shape[1]),3])
    # resize
    img = tf.image.resize(img,resize_size)
    
    # horizontal flip
    img = tf.image.random_flip_left_right(img)
    return img

# Dataloader
class ImageLabelFilelist(tf.data.Dataset):
    def _generator(img_root, input_list,idx_list):
        for item in zip(input_list,idx_list):
            yield (img_root + item[0], item[1])
    def __new__(cls, img_root,
                list_filename,
                filelist_reader=default_filelist_reader):
        img_list = filelist_reader(list_filename)
        classes = sorted(list(set([path.split('/')[0] for path in img_list])))
        classes_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_list = [classes_to_idx[l.split('/')[0]] for l in img_list]
        
        print("Data Loader")
        print("\tRoot: %s" % img_root)
        print("\tList: %s" % list_filename)
        print("\tNumber of classes: %d" % (len(classes)))
        
        return tf.data.Dataset.from_generator(
                cls._generator,
                output_types = (tf.string,tf.uint8),
                args = (img_root,img_list,idx_list,))

def get_dataset(data_loader_train,data_list_train,
                crop_size,resize_size,
                img_processor=default_img_processor,
                img_transformer=default_img_transformer):
    # set datasets we want.
    return ImageLabelFilelist(data_loader_train,data_list_train).shuffle(2000).take(2).map(img_processor)\
           .map(lambda x,y: (img_transformer(x,crop_size,resize_size),y))