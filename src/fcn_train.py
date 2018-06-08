import tensorflow as tf
import time
from datetime import datetime

############ DEBUG ################
from tensorflow.python import debug as tf_debug
############ DEBUG ################


import fcn
import fcn_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('debug', False,
                           """Debug flag indicator""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/ig02_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train ig02 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for ig02.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.

    with tf.device('/cpu:0'):
      images, labels = fcn_input.load_dataset(FLAGS.train_files, FLAGS.data_dir, FLAGS.batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = fcn.inference(images)

    # Calculate loss.
    loss = fcn.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = fcn.train(loss, global_step)

    # Agrego summary de imagenes
    tf.summary.image('input', images)
    tf.summary.image('label', labels[:,:,:,0:1])
    tf.summary.image('output autos', logits[:,:,:,0:1])
    tf.summary.image('output background', logits[:,:,:,1:2])

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    if FLAGS.debug: #borrar despues esto
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i, l = sess.run([logits, labels])
        print(i.shape)
        print(l.shape)
        exit()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      if FLAGS.debug:
        mon_sess = tf_debug.LocalCLIDebugWrapperSession(mon_sess)
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
    tf.app.run()
