import tensorflow as tf
import numpy as np

import mobilenetv2


def _model_fn(features, labels, mode, params):
    features = features['image']
    model = mobilenetv2.MobileNetV2(classes=10, data_format=params['data_format'])

    y_pred = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)
    predictions = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
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

    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=labels))
    metrics = {'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predictions)}

    if mode == tf.estimator.ModeKeys.TRAIN:
        boundaries = [100000, 200000]
        values = [0.01, 0.01, 0.001]
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(model.get_updates_for(features)):
            train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.EVAL
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=metrics,
    )


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    data_format = 'channels_first' if tf.test.is_gpu_available() else 'channels_last'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if data_format == 'channels_first':
        x_train = np.transpose(x_train, [0, 3, 1, 2])
        x_test = np.transpose(x_test, [0, 3, 1, 2])

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    def train_input_fn():
        return tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(lambda x, y: ({'image': tf.image.per_image_standardization(x)}, y)).repeat().shuffle(buffer_size=1000).batch(32).prefetch(8)

    def eval_input_fn():
        return tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(lambda x, y: ({'image': tf.image.per_image_standardization(x)}, y)).shuffle(buffer_size=1000).batch(4).prefetch(8)

    estimator = tf.estimator.Estimator(
        model_fn=_model_fn,
        model_dir='/tmp/mobilenetv2',
        params={'data_format': data_format},
        config=tf.estimator.RunConfig(
            save_checkpoints_secs=120,
        ),
    )

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            train_input_fn,
            max_steps=1000000,
        ),
        eval_spec=tf.estimator.EvalSpec(
            eval_input_fn,
            throttle_secs=120,
        )
    )

    input_shape = (None, 32, 32, 3)
    if data_format == 'channels_first':
        input_shape = (None, 3, 32, 32)
    inputs = tf.placeholder(tf.float32, shape=input_shape)
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': inputs,
    })
    estimator.export_savedmodel('/tmp/mobilenetv2', input_fn)


if __name__ == '__main__':
    main()