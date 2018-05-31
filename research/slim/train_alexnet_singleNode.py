import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import alexnet
from datasets import imagenet
from preprocessing import alexnet_preprocessing
import os

# ------ Global Variables -------
batch_size = 128
momentum = 0.9
train_dir =r'/work/projects/Project00755/logs/alexnet_single_node_02'
dataset_dir = r'/work/projects/Project00755/datasets/imagenet/tfrecords'
num_readers = 8
num_preprocessing_threads = 8
learning_rate = 0.01
weight_decay = 0.0005
validation_check = 20 # How often the validation accuracy should be calculated

# Set only the first GPU for Training
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def getImageBatchAndOneHotLabels(dataset_dir, dataset_name, num_readers, num_preprocessing_threads, batch_size):
    '''
    :param dataset_dir: directory where the tfrecord files are stored
    :param dataset_name: name of the dataset e.g. train / validation
    :return:
    '''
    dataset = imagenet.get_split(dataset_name, dataset_dir)
    # DataSetProvider on CPU
    with tf.device('/device:CPU:0'):
        # ------- Dataset Provider ---------
        provider_train = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=2 * batch_size,
            common_queue_min= batch_size)
        [image, label] = provider_train.get(['image', 'label'])

    # Preprocessing of Dataset
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
    return dataset, images, labels

graph = tf.Graph()
with graph.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset, images, labels = getImageBatchAndOneHotLabels(dataset_dir, 'train', num_readers, num_preprocessing_threads, batch_size)
    _, images_val, labels_val = getImageBatchAndOneHotLabels(dataset_dir, 'validation', 2, 2, batch_size)

    # Create Model network and endpoints

    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        logits, _ = alexnet.alexnet_v2(images, num_classes=dataset.num_classes)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            logits_val, _ = alexnet.alexnet_v2(images_val, num_classes=dataset.num_classes)


    #Metrics
    accuracy_validation = slim.metrics.accuracy(tf.to_int32(tf.argmax(logits_val, 1)),
                                                tf.to_int32(tf.argmax(labels_val, 1)))
    top5_accuracy = tf.metrics.mean(tf.nn.in_top_k(predictions=logits_val, targets=labels_val, k=5))

    # Added Loss Function
    tf.losses.softmax_cross_entropy(labels, logits)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')


def train_step_fn(session, *args, **kwargs):
    train_step = slim.learning.train_step
    total_loss, should_stop = train_step(session, *args, **kwargs)

    if train_step_fn.step % validation_check == 0:
        accuracy = session.run(train_step_fn.accuracy_validation)
        top5_accuracy = session.run(train_step_fn.top5_accuracy)
        print('Step %s - Loss: %.2f Val_Accuracy: %.2f%% Val_Top5_Accuracy %.2f%%' % (
        str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100, top5_accuracy * 100))


    train_step_fn.step += 1
    return [total_loss, should_stop]

train_step_fn.step = 0
train_step_fn.accuracy_validation = accuracy_validation
train_step_fn.top5_accuracy = top5_accuracy

slim.learning.train(
    train_tensor,
    logdir=train_dir,
    log_every_n_steps=10,
    summary_op=summary_op,
    train_step_fn=train_step_fn,
    save_summaries_secs=600,
    save_interval_secs=600,
    graph=graph
)


if __name__ == "__main__":
  tf.app.run()