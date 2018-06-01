import tensorflow
import fcn_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../data/ig02-cars/cars/',
"""Path to the ig02 data directory.""")

tf.app.flags.DEFINE_string('filelist_dir', '../data/ig02-cars/',
"""Path to the ig02 file lists directory.""")

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 177
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 46

def _create_weights(name, shape, stddev, wd=None):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, name=name))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _create_bias(name, shape):
    return tf.Variable(name=name, tf.constant(1., shape=shape, dtype=tf.float32))

def inference(images):
    """FCN model building.

    Args:
      images: Input images

    Returns:
      Logits.
    """

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _create_weights(
            'weights',
            shape=[3, 3, 1, 16],
            stddev=5e-2,
            wd=None
        )
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
        biases = _create_bias('biases', [16])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # TODO: summary

    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1,3,3,1],
        strides=[1,2,2,1],
        padding='SAME',
        name='pool1'
    )

    # norm1
    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #    name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _create_weights(
            'weights',
            shape=[3,3,16,32],
            stddev=5e-2,
            wd=None
        )
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
        biases = _create_bias('biases', [32])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # TODO: summary

    pool2 = tf.nn.max_pool(
        conv2,
        ksize=[1,3,3,1],
        strides=[1,2,2,1],
        padding='SAME',
        name='pool2'
    )

    # replacing fully connecteds

    # conv1_1
    with tf.variable_scope('conv1_1') as scope:
        kernel = _create_weights(
            'weights',
            shape=[1,1,32,32],
            stddev=5e-2,
            wd=0.004
        )
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
        biases = _create_bias('biases', tf.shape(conv)[0])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(pre_activation, name=scope.name)
        # TODO: summary

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
        kernel = _create_weights(
            'weights',
            shape=[1,1,32,NUM_CLASSES],
            stddev=1/32.0,
            wd=None
        )
        conv = tf.nn.conv2d(conv1_1, kernel, [1,1,1,1], padding='SAME')
        biases = _create_bias('biases', tf.shape(conv)[0])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_1 = pre_activation # no se le aplica softmax aca
        # TODO: summary

    return conv2_1

def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entroy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_sumaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    loss_averages_op = _add_loss_sumaries(total_loss)

    with tf.control_dependecies([loss_averages_op]):
        opt = tf.train.AdamOptimizer() # default values
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op
