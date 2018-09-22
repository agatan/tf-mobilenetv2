import tensorflow as tf
import numpy as np
from tqdm import tqdm

import mobilenetv2


def main():
    config = tf.ConfigProto(log_device_placement=True)
    tf.enable_eager_execution(config=config)

    data_format = 'channels_first' if tf.test.is_gpu_available() else 'channels_last'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if data_format == 'channels_first':
        x_train = np.transpose(x_train, [0, 3, 1, 2])
        x_test = np.transpose(x_test, [0, 3, 1, 2])

    x_train = x_train.astype(np.float32)
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = x_test.astype(np.float32)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = mobilenetv2.MobileNetV2(classes=10, data_format=data_format)
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()
    checkpointer = tf.train.Checkpoint(optimizer=optimizer, model=model, optimizer_step=global_step)
    latest_path = tf.train.latest_checkpoint('/tmp/mobilenetv2_eager')
    checkpointer.restore(latest_path)
    print(latest_path)

    for epoch in range(100):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        for x, y in tqdm(dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y))
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables), global_step=global_step)
        print(checkpointer.save(file_prefix='/tmp/mobilenetv2_eager/ckpt'))

        eval_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(4)
        losses = []
        accuracies = []
        for x, y in eval_dataset:
            y_pred = model(x, training=False)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y))
            accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred=y_pred, y_true=y))
            losses.append(loss.numpy())
            accuracies.append(accuracy.numpy())
        losses = np.array(losses)
        accuracies = np.array(accuracies)
        print('Loss: {:.3f}'.format(np.mean(losses)))
        print('Accuracy: {:.3f}%'.format(np.mean(accuracies)))


if __name__ == '__main__':
    main()
