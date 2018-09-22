import tensorflow as tf
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm

import mobilenetv2


def make_dataset(x, y, batch_size, data_format):
    def g(x, y):
        aug = iaa.Sequential([
            iaa.CropAndPad(px=(-4, 4)),
            iaa.Fliplr(0.5),
        ])
        indices = np.random.permutation(len(x))
        x = x[indices]
        y = y[indices]
        n_batches = (len(x) - 1) // batch_size + 1
        for n in range(n_batches):
            s = n * batch_size
            e = min((n + 1) * batch_size, len(x))
            xs = x[s:e]
            ys = y[s:e]
            xs = aug.augment_images(xs).astype(np.float32)
            xs -= np.mean(xs, axis=(1, 2, 3), keepdims=True)
            xs /= (np.std(xs, axis=(1, 2, 3), keepdims=True))
            if data_format == 'channels_first':
                xs = np.transpose(xs, [0, 3, 1, 2])
            yield {'image': xs}, ys
    if data_format == 'channels_first':
        output_shapes = ({'image': tf.TensorShape((None, 3, 32, 32))}, tf.TensorShape((None, None)))
    else:
        output_shapes = ({'image': tf.TensorShape((None, 32, 32, 3))}, tf.TensorShape((None, None)))
    return tf.data.Dataset.from_generator(lambda: g(x, y), output_types=({'image': tf.float32}, tf.float32), output_shapes=output_shapes).prefetch(32)


def main():
    config = tf.ConfigProto(log_device_placement=True)
    tf.enable_eager_execution(config=config)

    data_format = 'channels_first' if tf.test.is_gpu_available() else 'channels_last'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    model = mobilenetv2.MobileNetV2(classes=10, data_format=data_format)
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()
    checkpointer = tf.train.Checkpoint(optimizer=optimizer, model=model, optimizer_step=global_step)
    latest_path = tf.train.latest_checkpoint('/tmp/mobilenetv2_eager')
    checkpointer.restore(latest_path)
    print(latest_path)

    for epoch in range(100):
        dataset = make_dataset(x_train, y_train, batch_size=32, data_format=data_format)
        for x, y in tqdm(dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x['image'], training=True)
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y))
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables), global_step=global_step)
        print(checkpointer.save(file_prefix='/tmp/mobilenetv2_eager/ckpt'))

        eval_dataset = make_dataset(x_test, y_test, batch_size=8, data_format=data_format)
        losses = []
        accuracies = []
        for x, y in eval_dataset:
            y_pred = model(x['image'], training=False)
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
