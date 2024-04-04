import importlib
import pathlib
import argparse
import time

import silence_tensorflow.auto

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow_datasets as tfds

# Global Config
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)
VERBOSITY = 0

MODELS_DIR = pathlib.Path(f'{pathlib.Path.cwd()}/generated/baselines/')
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocess, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype = tf.float32)
        image = preprocess(image)
        return image, label


    (test_set), ds_info = tfds.load(
        name=dataset,
        split='test',
        with_info=True,
        as_supervised=True)

    test_set = test_set.map(map_func=resize_image)
    test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)

    num_of_output = ds_info.features['label'].num_classes

    return (test_set, num_of_output)


def setup_model(model, model_name, test_set, num_of_output):
    base_architecture = model(
        include_top=False,
        input_shape=IMAGE_SIZE + (3, ),
        pooling='avg'
    )
    base_architecture.trainable = False
    outputs = tf.keras.layers.Dense(num_of_output, activation="softmax")(base_architecture.output)

    model = tf.keras.Model(base_architecture.input, outputs)
    model.load_weights(f'{pathlib.Path.cwd()}/generated/baselines/{args.model_name}/run_1')
    
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def measure_latency(model, test_set):
    warmup_set = test_set.take(10)
    inference_set = test_set.take(1000)

    with tf.device('/cpu:0'):
        print('CPU Latency')
        # warmup, measurement not taken here 
        model.evaluate(warmup_set, verbose=0)
        # inference
        model.evaluate(inference_set, verbose=2)

    with tf.device('/gpu:0'):
        print('\nGPU Latency')
        # warmup, measurement not taken here 
        model.evaluate(warmup_set, verbose=0)
        # inference
        model.evaluate(inference_set, verbose=2)


def main(args):
    module = prepare_module(args.module_name)
    model = getattr(module, args.model_name)
    preprocess_input = getattr(module, 'preprocess_input')

    (test_set, num_of_ouput) = prepare_dataset(
        dataset='oxford_flowers102', 
        preprocess=preprocess_input, 
        batch_size=1)
    
    model = setup_model(
        model=model, 
        model_name=args.model_name, 
        test_set=test_set, 
        num_of_output=num_of_ouput)

    measure_latency(model, test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--module_name')
    parser.add_argument('--model_name')

    args = parser.parse_args()

    main(args)
