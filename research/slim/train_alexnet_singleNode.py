import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import alexnet
from datasets import imagenet
from preprocessing import alexnet_preprocessing
import os


# ------ Global Variables -------
batch_size = 32
train_dir =r'/work/projects/Project00755/logs/alexnet_single_node_02'
dataset_dir = r'/work/projects/Project00755/datasets/imagenet/tfrecords'
num_readers = 8
num_preprocessing_threads = 8

# Set only the first GPU for Training
os.environ["CUDA_VISIBLE_DEVICES"]="0"

with tf.Graph().as_default():

    dataset = imagenet.get_split('train', dataset_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    # DataSetProvider on CPU
    with tf.device('/device:CPU:0'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=20 * batch_size,
            common_queue_min=10 * batch_size)
        [image, label] = provider.get(['image', 'label'])

    # Preprocess Imagenet Pictures
    train_image_size =alexnet.alexnet_v2.default_image_size
    image = alexnet_preprocessing.preprocess_image(image, train_image_size, train_image_size)

    # Generate Batches
    images, labels = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocessing_threads,
          capacity=5 * batch_size)
    labels = slim.one_hot_encoding(
          labels, dataset.num_classes)

    # Create Model network and endpoints
    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        logits, end_points = alexnet.alexnet_v2(images, num_classes=dataset.num_classes)

    # Added Loss Function
    tf.losses.softmax_cross_entropy(labels, logits)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    train_tensor = slim.learning.create_train_op(total_loss, optimizer)




    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        log_every_n_steps=10,
        summary_op=summary_op,
        save_summaries_secs=600,
        save_interval_secs=600
    )


if __name__ == "__main__":
  tf.app.run()