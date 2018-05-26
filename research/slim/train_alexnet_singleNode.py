import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.alexnet import alexnet_v2
from datasets import imagenet

# ------ Global Variables -------
batch_size = 32
train_dir =r'/work/projects/Project00755/logs/alexnet_single_node_02'
dataset_dir = r'/work/projects/Project00755/datasets/imagenet/tfrecords'
num_readers = 4


#--------- Program --------
tf.logging.set_verbosity(tf.logging.INFO)

# Read data from TFRecords

dataset = imagenet.get_split('train', dataset_dir)

with tf.device('/device:CPU:0'):
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])



net, end_points = alexnet_v2(input)


if __name__ == "__main__":
  tf.app.run()