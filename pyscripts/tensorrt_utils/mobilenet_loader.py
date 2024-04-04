import importlib
import tensorflow_datasets as tfds
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1
IMAGE_SIZE = (224, 224)


def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocess, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype = tf.float32)
        return image, label


    (training_set), ds_info = tfds.load(
        name=dataset,
        split='train',
        with_info=True,
        as_supervised=True)

    training_set = training_set.map(map_func=resize_image)
    training_set = training_set \
        .shuffle(256) \
        .batch(batch_size) \
        .prefetch(AUTOTUNE)

    return training_set


def load_data():
    module_name = 'mobilenet_v3'
    module = prepare_module(module_name)
    preprocess_input = getattr(module, 'preprocess_input')

    training_set = prepare_dataset('oxford_flowers102', preprocess_input, 1)

    for image, label in training_set.take(128):
        yield { 'input_1': image.numpy() }
