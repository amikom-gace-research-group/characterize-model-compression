# Setup
import silence_tensorflow.auto

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import argparse
import importlib
import psutil
import tensorflow_model_optimization as tfmot
import numpy as np
import tensorflow_datasets as tfds
import pathlib
from datetime import datetime
import time
import tempfile

from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry

# Global configs
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)
VERBOSITY = 2
FINE_TUNING_EPOCHS = 5

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


    (training_set, validation_set, test_set), ds_info = tfds.load(
        name=dataset,
        split=['train', 'validation', 'test'],
        with_info=True,
        as_supervised=True)

    training_set = training_set.map(map_func=resize_image)
    training_set = training_set.shuffle(256) \
        .batch(batch_size) \
        .prefetch(AUTOTUNE)

    # val_data
    validation_set = validation_set.map(map_func=resize_image, num_parallel_calls=AUTOTUNE)
    validation_set = validation_set.batch(batch_size=batch_size).prefetch(AUTOTUNE)

    test_set = test_set.map(map_func=resize_image)
    test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)

    return (training_set, validation_set, test_set)


def setup_model(model, model_name, run_num):
    base_architecture = model(
        include_top=False,
        input_shape=IMAGE_SIZE + (3, ),
        pooling='avg',
    )
    base_architecture.trainable = False
    outputs = tf.keras.layers.Dense(102, activation="softmax")(base_architecture.output)

    model = tf.keras.Model(base_architecture.input, outputs)
    model.load_weights(
        f'{pathlib.Path.cwd()}/generated/baselines/{args.model_name}/run_{args.run_num}')
    
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    return model


def prune(model_name, model, training_set, validation_set, test_set, target_sparsity, saving_dir):
    def set_pruning_params():
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity, 
                begin_step=0)
        }

        return pruning_params


    def apply_pruning(layer):
        pruning_params = set_pruning_params()
        if isinstance(layer, prunable_layer.PrunableLayer) or \
            hasattr(layer, 'get_prunable_weights') or \
            prune_registry.PruneRegistry.supports(layer):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

        return layer


    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning)

    model_for_pruning.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    
    # print(f'\n{tuner} Process!')
    print(f'Begins at {datetime.now()}')

    start_time = time.time()
    model_for_pruning.fit(
        training_set,
        epochs=FINE_TUNING_EPOCHS, 
        validation_data=validation_set,
        verbose=VERBOSITY,
        callbacks=callbacks)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f'Pruning took {round(elapsed, 2)}s!')

    _, pruned_accuracy = model_for_pruning.evaluate(test_set, verbose=VERBOSITY)
    print(f'Pruned accuracy: {round(pruned_accuracy * 100, 2)}')

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    tf.saved_model.save(model_for_export, saving_dir)
    print('Saved pruned Keras model to:', saving_dir)


def create_saving_dir(model_name, target_sparsity, run_num):
    base_dir=f"{pathlib.Path.cwd()}/generated/pruned_keras"

    model_dir = f"{model_name}_{target_sparsity}_sparsity/run_{run_num}"
    saving_dir = pathlib.Path(f'{base_dir}/{model_dir}')

    saving_dir.mkdir(exist_ok=True, parents=True)
    return saving_dir


def main(args):
    module = prepare_module(args.module_name)
    model = getattr(module, args.model_name)
    preprocess_input = getattr(module, 'preprocess_input')

    (training_set, validation_set, test_set) = prepare_dataset(
        dataset='oxford_flowers102', 
        preprocess=preprocess_input, 
        batch_size=4)

    saving_dir = create_saving_dir(
        model_name=args.model_name,
        target_sparsity=args.target_sparsity,
        run_num=args.run_num)

    if args.tuner == "CPU":
        with tf.device('/cpu:0'):
            orig_model = setup_model(model, args.model_name, args.run_num)
            print(f'Loaded {args.model_name}, run {args.run_num}')
            pruned_keras_folder = prune(
                model_name=args.model_name,
                model=orig_model,
                training_set=training_set,
                validation_set=validation_set,
                test_set=test_set, 
                target_sparsity=float(args.target_sparsity), 
                saving_dir=saving_dir)
    else:
        orig_model = setup_model(model, args.model_name, args.run_num)
        print(f'Loaded {args.model_name}, run {args.run_num}')
        pruned_keras_folder = prune(
            model_name=args.model_name,
            model=orig_model,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set, 
            target_sparsity=float(args.target_sparsity), 
            saving_dir=saving_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--module_name')
    parser.add_argument('--model_name')
    parser.add_argument('--target_sparsity')
    parser.add_argument('--run_num')
    parser.add_argument('--tuner')

    args = parser.parse_args()

    main(args)
