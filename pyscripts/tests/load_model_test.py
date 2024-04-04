import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_datasets as tfds


# Configs
AUTOTUNE=tf.data.AUTOTUNE
IMAGE_SIZE=(224, 224)


def prepare_dataset(dataset, batch_size):
    def resize_image(image, label):
        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.cast(image, dtype = tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label


    (test_set), ds_info = tfds.load(
        name=dataset,
        split='test',
        with_info=True,
        as_supervised=True)

    test_set = test_set.map(map_func=resize_image)
    test_set = test_set.batch(batch_size).prefetch(AUTOTUNE)

    return test_set


def setup_model():
    base_architecture = tf.keras.applications.EfficientNetB5(
        include_top=False,
        input_shape=(IMAGE_SIZE) + (3, ),
        pooling='avg'
    )

    base_architecture.trainable = False
    outputs = tf.keras.layers.Dense(102, activation="softmax")(base_architecture.output)

    model = tf.keras.Model(base_architecture.input, outputs)

    model.load_weights('generated/baselines/EfficientNetB5/run_1')
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def main():
    model = setup_model()
    test_set = prepare_dataset('oxford_flowers102', 4)

    model.evaluate(test_set)


if __name__ == "__main__":
    main()