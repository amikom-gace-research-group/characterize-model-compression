import silence_tensorflow.auto
import argparse
import importlib
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow_datasets as tfds
import numpy as np
import pathlib
import time
from datetime import datetime


# Global configs
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)


# Dataset Prep
def prepare_module(module_name):
    module = importlib.import_module(f'keras.applications.{module_name}')
    return module


def prepare_dataset(dataset, preprocess, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype = tf.float32)
        image = preprocess(image)
        return image, label


    (training_set), ds_info = tfds.load(
        name=dataset,
        split='train',
        with_info=True,
        as_supervised=True)

    training_set = training_set.map(map_func=resize_image)
    training_set = training_set.shuffle(256).batch(batch_size).prefetch(AUTOTUNE)

    return training_set


# Convert WITH float16 post training quant
# 2X Size Difference!
def quantize_to_float16(model_name, model, saving_dir):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    start_time = time.time()
    print(f'starts float16 conversion at {datetime.now()}')
    float16_postq = converter.convert()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Finish float16 conversion in {round(elapsed, 2)}s!\n')

    float16_postq_file = saving_dir / f'./PQ_{model_name}_float16.tflite'
    float16_postq_file.write_bytes(float16_postq)

    return float16_postq_file


# Convert WITH dynamic range post training quant
# 4X Size Difference!
def quantize_to_dynamic(model_name, model, saving_dir):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    start_time = time.time()
    print(f'starts dynamic range conversion at {datetime.now()}')
    dynamic_postq = converter.convert()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Finish dynamic range conversion in {round(elapsed, 2)}s!\n')

    dynamic_postq_file = saving_dir / f'./PQ_{model_name}_dynamic_range.tflite'
    dynamic_postq_file.write_bytes(dynamic_postq)

    return dynamic_postq_file


# Full Integer Quantization
def quantize_to_full_int(model_name, model, training_set, saving_dir):
    def representative_data_gen():
        for batch in training_set.take(128):
            yield [batch[0]]


    # Convert WITH full integer post training quant
    # 4X Size Difference!
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Try to change to normal ints! or floats!
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    start_time = time.time()
    print(f'starts full int conversion at {datetime.now()}')
    full_int_postq = converter.convert()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Finish full int conversion in {round(elapsed, 2)}s!\n')

    full_int_postq_file = saving_dir / f'./PQ_{model_name}_full_integer.tflite'
    full_int_postq_file.write_bytes(full_int_postq)

    return full_int_postq_file 


def main(args):
    saving_dir = pathlib.Path(f"{pathlib.Path.cwd()}/generated/ptq_tflite/DEF_PQ_{args.model_name}/run_{args.run_num}")
    saving_dir.mkdir(exist_ok=True, parents=True)

    module = prepare_module(args.module_name)
    preprocess_input = getattr(module, 'preprocess_input')

    (training_set) = prepare_dataset(
        dataset='oxford_flowers102', 
        preprocess=preprocess_input, 
        batch_size=1)

    saved_model = f'{pathlib.Path.cwd()}/generated/baselines/{args.model_name}/run_{args.run_num}' 
    model = tf.saved_model.load(saved_model)

    quantize_to_float16(
        model_name=args.model_name,
        model=model, 
        saving_dir=saving_dir)

    quantize_to_dynamic(
        model_name=args.model_name,
        model=model, 
        saving_dir=saving_dir)

    quantize_to_full_int(
        model_name=args.model_name,
        model=model,
        training_set=training_set,
        saving_dir=saving_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--module_name')
    parser.add_argument('--model_name')
    parser.add_argument('--run_num')

    args = parser.parse_args()

    main(args)
