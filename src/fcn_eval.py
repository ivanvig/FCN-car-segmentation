'''
Architecture based on: Fully Convolutional Networks for Semantic Segmentation
https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

Code based on: Tensorflow's Convoltional Neural Networks tutorial
https://www.tensorflow.org/tutorials/deep_cnn
'''

from datetime import datetime
import math
import time
import numpy as np

import tensorflow as tf
import fcn
import fcn_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/ig02_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/ig02_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often to run the eval in secs.""")
tf.app.flags.DEFINE_integer('num_examples', 46,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
"""Whether to run eval only once.""")

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      # TODO: Implement IoU insetad of this
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval IG-02 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for ig02.
    if FLAGS.eval_data == 'test':
        files = FLAGS.eval_files
    elif FLAGS.eval_data == 'train_eval':
        files = FLAGS.train_files
    else:
        raise ValueError

    images, labels = fcn_input.load_dataset(
      files,
      FLAGS.data_dir,
      FLAGS.batch_size
    )

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = fcn.inference(images)

    # Calculate predictions.
    _, mask = tf.nn.top_k(logits)

    # se invierte la mascara
    mask = tf.subtract(tf.fill(tf.shape(mask), 1.0), tf.cast(mask, tf.float32))

    top_k_op = tf.reduce_mean(
      tf.cast(tf.equal(mask, labels[:,:,:,0:1]), tf.float32),
      [1,2]
    )

    # Adding images to summary
    tf.summary.image('input', images)
    tf.summary.image('label', labels[:,:,:,0:1])
    tf.summary.image('Mask', mask[:,:,:,0:1])

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        fcn.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  # care: this deletes previous checkpoints if the dir exists
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
