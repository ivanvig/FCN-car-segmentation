import tensorflow as tf

def load_dataset(filename, data_path, batch_size=1):
    image_names = tf.constant(open(filename).read().split('\n'))
    dataset = tf.data.Dataset.from_tensor_slices(image_names)
    dataset = dataset.map(lambda x: _parse_function(x, data_path))

    #batch = 1 porque tf no puede hacer batch de imagenes de distintas dimensiones
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def _parse_function(imagename, data_path):
    img = tf.read_file(data_path+imagename+".image.png") # TODO: usar os
    img_dec = tf.image.decode_image(img, channels=3)
    img = tf.image.rgb_to_grayscale(img_dec)
    img_std = tf.image.per_image_standardization(img)

    label = tf.image.decode_png(tf.read_file(data_path+imagename+".mask.png"), channels=1)
    label = tf.divide(label,255)

    return img_std, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    images, labels = load_dataset("../data/ig02-cars/cars_train.txt", "../data/ig02-cars/cars/")
    with tf.Session() as sess:
        i, l = sess.run([images, labels])
        #plt.imshow(i[0,:,:,0], cmap="gray")
        plt.imshow(l[0,:,:,0], cmap="gray")
        plt.show()
