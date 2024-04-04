import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import pathlib
import importlib
import time
from datetime import datetime


# General Configs
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)


# Dataset Prep
def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocess, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype=tf.float32)
        image = preprocess(image)
        return image, label


    (training_set), ds_info = tfds.load(
        name=dataset,
        split='train',
        with_info=True,
        as_supervised=True)

    training_set = training_set.map(map_func=resize_image)
    training_set = training_set.shuffle(256) \
        .batch(batch_size) \
        .prefetch(AUTOTUNE)

    return training_set


def quantize_pruned_model(pruned_model, training_set, saving_dir):
    def representative_dataset_gen():
        for batch in training_set.take(128):
            yield [batch[0]]


    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter.representative_dataset = representative_dataset_gen

    start_time = time.time()
    print(f'\nQuantization started at {datetime.now()}')
    pruned_quantized_model = converter.convert()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f'Quantization took {round(elapsed, 2)}s!')

    pruned_quantized_file = f'{saving_dir}/pruned_quantized.tflite'
    with open(pruned_quantized_file, 'wb') as f:
        f.write(pruned_quantized_model)
        print('Saved quantized and pruned TFLite model to:', pruned_quantized_file)

    return pruned_quantized_file


def create_saving_dir(model_name, sparsity, run_num):
    base_dir = f'{pathlib.Path.cwd()}/generated/pruned_tflite'
    model_dir = f'{model_name}_{sparsity}_sparsity/run_{run_num}'

    saving_dir = pathlib.Path(f'{base_dir}/{model_dir}')
    saving_dir.mkdir(exist_ok=True, parents=True)

    return saving_dir


def main(args):
    module = prepare_module(args.module_name)
    preprocess_input = getattr(module, 'preprocess_input')

    training_set = prepare_dataset(
            dataset='oxford_flowers102',
            preprocess=preprocess_input,
            batch_size=1)

    base_model = f'{pathlib.Path.cwd()}/generated/pruned_keras'
    saved_model = f'{args.model_name}_{args.target_sparsity}_sparsity/run_{args.run_num}' 
    model = tf.saved_model.load(f'{base_model}/{saved_model}')
    
    print(f"Loaded {base_model}/{saved_model}")

    quantize_pruned_model(
        pruned_model=model,
        training_set=training_set,
        saving_dir=create_saving_dir(args.model_name, args.target_sparsity, args.run_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--module_name')
    parser.add_argument('--model_name')
    parser.add_argument('--target_sparsity')
    parser.add_argument('--run_num')

    args = parser.parse_args()

    main(args)
