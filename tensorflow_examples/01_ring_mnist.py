# Source: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras

import json
import os
import uuid

import numpy as np
import tensorflow as tf
import valohai

# Populate "TF_CONFIG" environment variable with the cluster configuration
# and identity of this worker.
primary_local_ips = [m.primary_local_ip for m in valohai.distributed.members()]
worker_addresses = [f'{ip}:12345' for ip in primary_local_ips]
tf_config = {
    'cluster': {
        'worker': worker_addresses
    },
    'task': {'type': 'worker', 'index': valohai.distributed.rank}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)

# "MultiWorkerMirroredStrategy" should be instantiated before any code that might generate "ops",
# preferably at the start of the program.
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#train_the_model
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.RING,
    ),
)


def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(60000).repeat().batch(batch_size)
    return train_dataset


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'],
    )
    return model


def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])


per_worker_batch_size = 64
num_workers = valohai.distributed.required_count
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metadata)
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70, callbacks=[callback])

# Typically, only the model saved by the chief should be referenced for restoring or serving.
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#model_saving_and_loading
if valohai.distributed.me().is_master:
    suffix = uuid.uuid4()
    output_path = valohai.outputs().path(f'model-{suffix}.h5')
    multi_worker_model.save(output_path)
