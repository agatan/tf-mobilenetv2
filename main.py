import tensorflow as tf
import numpy as np

import mobilenetv2


def main():
    tf.enable_eager_execution()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)
    y_test = tf.keras.utils.to_categorical(y_test)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

    model = mobilenetv2.MobileNetV2(classes=10)
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()
    checkpointer = tf.train.Checkpoint(optimizer=optimizer, model=model, optimizer_step=global_step)
    latest_path = tf.train.latest_checkpoint('/tmp/mobilenetv2_eager')
    checkpointer.restore(latest_path)
    print(latest_path)

    for x, y in dataset:
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=global_step)
        print(checkpointer.save(file_prefix='/tmp/mobilenetv2_eager/ckpt'))

    eval_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(4)
    for x, y in eval_dataset:
        y_pred = model(x, training=False)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y))
        accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_pred=y_pred, y_true=y))
        print('Loss: {:.3f}'.format(loss))
        print('Accuracy: {:.3f}%'.format(accuracy))


if __name__ == '__main__':
    main()