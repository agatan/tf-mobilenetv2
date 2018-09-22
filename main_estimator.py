import tensorflow as tf
import numpy as np

import mobilenetv2


def _model_fn(features, labels, mode, params):
    features = features['image']
    model = mobilenetv2.MobileNetV2(classes=10, data_format=params['data_format'])
    print(mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        y_pred = model(features, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=labels))
        with tf.control_dependencies(model.get_updates_for(features)):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)
    y_pred = model(features, training=False)
    predictions = tf.argmax(y_pred, axis=1)
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=labels))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions)},
        )
    result = {
        'classes': predictions,
        'probabilities': y_pred,
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(result),
        }
    )


def main():
    data_format = 'channels_first' if tf.test.is_gpu_available() else 'channels_last'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if data_format == 'channels_first':
        x_train = np.transpose(x_train, [0, 3, 1, 2])
        x_test = np.transpose(x_test, [0, 3, 1, 2])

    x_train = x_train.astype(np.float32)
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = x_test.astype(np.float32)
    y_test = tf.keras.utils.to_categorical(y_test)

    def train_input_fn():
        return tf.data.Dataset.from_tensor_slices(({'image': x_train}, y_train)).batch(32)

    def eval_input_fn():
        return tf.data.Dataset.from_tensor_slices(({'image': x_test}, y_test)).batch(4)

    estimator = tf.estimator.Estimator(
        model_fn=_model_fn,
        model_dir='/tmp/mobilenetv2',
        config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=True)),
        params={'data_format': data_format},
    )
    estimator.train(train_input_fn)
    estimator.evaluate(eval_input_fn)

    input_shape = (None, 28, 28, 1)
    inputs = tf.placeholder(tf.float32, shape=input_shape)
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': inputs,
    })
    estimator.export_savedmodel('/tmp/mobilenetv2', input_fn)


if __name__ == '__main__':
    main()