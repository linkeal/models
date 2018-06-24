import argparse
import sys, os

import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from nets import alexnet
from datasets import imagenet
from preprocessing import alexnet_preprocessing

# ------ Global Variables -------
batch_size = 128
momentum = 0.9
train_dir =r'/work/projects/Project00755/logs/alexnet_distributed/'
root_dataset_dir = r'/work/projects/Project00755/datasets/imagenet/tfrecords/2_worker/'
num_readers = 8
num_preprocessing_threads = 8
learning_rate = 0.01
#weight_decay = 0.0005

FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

        # ------- READING DATA ---------
        dataset_dir=root_dataset_dir + 'worker_' + str(FLAGS.task_index)
        print('Worker {} is reading files from {}'.format(str(FLAGS.task_index), dataset_dir))
        dataset = imagenet.get_split('train', dataset_dir)

        provider_train = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=2 * batch_size,
            common_queue_min=batch_size)
        [image, label] = provider_train.get(['image', 'label'])

        # Preprocessing of Dataset
        train_image_size = alexnet.alexnet_v2.default_image_size
        image = alexnet_preprocessing.preprocess_image(image, train_image_size, train_image_size)

        # Generate Batches
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocessing_threads,
            capacity=5 * batch_size)
        labels = slim.one_hot_encoding(
            labels, dataset.num_classes)


        # Build model
        logits, _ = alexnet.alexnet_v2(images, num_classes=dataset.num_classes)
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        global_step = tf.contrib.framework.get_or_create_global_step()


        train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)\
            .minimize(loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)