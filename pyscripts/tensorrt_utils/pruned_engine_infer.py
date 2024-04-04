import pathlib
import importlib
import argparse
import time
from datetime import datetime

import silence_tensorflow.auto

import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

import tensorflow as tf
import tensorflow_datasets as tfds

# Global config
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)


def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocessor, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype=tf.float32)
        image = preprocessor(image)

        return image, label


    (test_set), ds_info = tfds.load(
        dataset,
        split='test',
        with_info=True,
        as_supervised=True)

    test_set = test_set.map(map_func=resize_image, num_parallel_calls=AUTOTUNE)
    test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)

    return test_set


def do_inference(path: str, test_set):
    load_engine = EngineFromBytes(BytesFromPath(path))

    true_labels = []
    predictions = []

    print(f"\nStarts engine loading at {datetime.now()}")
    with TrtRunner(load_engine) as runner:
        print(f"Engine loaded at {datetime.now()}")

        start_time = time.time()
        for image, label in test_set:
            true_labels.extend(label.numpy())

            outputs = runner.infer(feed_dict={'input_1': image.numpy()})
            inferred = np.argmax(outputs['dense'])

            predictions.append(inferred)

        end_time = time.time()

        elapsed = round(end_time - start_time, 2)
        print(f"Inference done in {elapsed} seconds!")

    accuracy = np.mean(np.array(true_labels) == np.array(predictions))
    print(f"Model accuracy: {round(accuracy * 100, 2)}")


def main(args):
    module = prepare_module(args.module_name)
    preprocess_input = getattr(module, 'preprocess_input')

    base_dir = f'{pathlib.Path.cwd()}/generated/pruned_tensorrt'
    int8_file_loc = f'{args.model_name}_{args.sparsity}_sparsity.engine'
    full_engine_loc=f'{base_dir}/{int8_file_loc}'

    print(f'INT8 Path: {full_engine_loc}')

    test_set = prepare_dataset(
        dataset='oxford_flowers102',
        preprocessor=preprocess_input,
        batch_size=1
    )

    do_inference(full_engine_loc, test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--module_name')
    parser.add_argument('--model_name')
    parser.add_argument('--sparsity')

    args = parser.parse_args()

    main(args)
