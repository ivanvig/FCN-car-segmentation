import tensorflow as tf
from distutils.version import LooseVersion

if LooseVersion(tf.__version__) < LooseVersion("1.5"):
    setattr(tf, 'data', tf.contrib.data)

def load_dataset(filename, data_path, batch_size=1):
    # ese -1 es para eliminar un elemento vacio de la lista
    image_names = tf.constant(open(filename).read().split('\n')[:-1])
    dataset = tf.data.Dataset.from_tensor_slices(image_names)
    dataset = dataset.map(lambda x: _parse_function(x, data_path))

    dataset = dataset.shuffle(100).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def _parse_function(imagename, data_path):
    img = tf.read_file(data_path+imagename+".image.png") # TODO: usar os
    img_dec = tf.image.decode_png(img, channels=3)
    img = tf.image.rgb_to_grayscale(img_dec)
    # submuestreo y pongo todo a la misma dimension para usar batch > 1
    #img = tf.image.resize_images(img, [240, 380]) # TODO: hacerlo dinamico
    img = tf.image.resize_images(img, [120, 160]) # TODO: hacerlo dinamico
    img_std = tf.image.per_image_standardization(img)

    label = tf.image.decode_png(tf.read_file(data_path+imagename+".mask.png"), channels=1)
    label = tf.image.resize_images(label, [120, 160]) # TODO: hacerlo dinamico
    label = tf.divide(label,255)
    label_bg = tf.subtract(tf.fill(tf.shape(label), 1.0), label)

    return img_std, tf.concat([label, label_bg], 2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    images, labels = load_dataset("../data/ig02-cars/cars_train.txt", "../data/ig02-cars/cars/")
    with tf.Session() as sess:
        i,l = sess.run([images, labels])
        plt.imshow(i[0,:,:,0], cmap="gray")
        plt.imshow(l[0,1,:,:,0], cmap="gray")
        plt.show()
