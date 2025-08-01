"""Training pipeline module for personality prediction using TFX."""

import os
import tensorflow as tf
import tensorflow_transform as tft
from kerastuner import HyperParameters
from tfx.components.trainer.fn_args_utils import FnArgs

# Label target
LABEL_KEY = "Personality"

# Daftar fitur kategorikal & numerik
CATEGORICAL_FEATURE_KEYS = [
    "Stage_fear",
    "Drained_after_socializing"
]

NUMERIC_FEATURE_KEYS = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]

def transformed_name(key: str) -> str:
    """Generate transformed feature name."""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Read TFRecord files with GZIP compression."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs=10, batch_size=64):
    """Create input dataset for training/evaluation."""
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
        num_epochs=num_epochs,
        shuffle=True
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

def model_builder(hp=None):
    """Build Keras model for personality classification."""
    inputs = {}
    encoded_inputs = []

    for key in CATEGORICAL_FEATURE_KEYS:
        feat_key = transformed_name(key)
        inputs[feat_key] = tf.keras.Input(shape=(1,), name=feat_key, dtype=tf.int64)
        embed = tf.keras.layers.Embedding(input_dim=10, output_dim=4)(inputs[feat_key])
        flat = tf.keras.layers.Flatten()(embed)
        encoded_inputs.append(flat)

    for key in NUMERIC_FEATURE_KEYS:
        feat_key = transformed_name(key)
        inputs[feat_key] = tf.keras.Input(shape=(1,), name=feat_key, dtype=tf.float32)
        encoded_inputs.append(inputs[feat_key])

    x = tf.keras.layers.Concatenate()(encoded_inputs)

    units_1 = hp.Int("units_1", 32, 128, step=16) if hp else 64
    units_2 = hp.Int("units_2", 16, 128, step=16) if hp else 32
    learning_rate = hp.Choice("learning_rate", [0.001, 0.01, 0.1]) if hp else 0.001

    x = tf.keras.layers.Dense(units_1, activation='relu')(x)
    x = tf.keras.layers.Dense(units_2, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Wrap model for TensorFlow Serving."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    """TFX entry point for training."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    model = (model_builder(hp=HyperParameters.from_config(fn_args.hyperparameters))
             if fn_args.hyperparameters else model_builder())

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs'))

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=100,
        validation_steps=50,
        epochs=5,
        callbacks=[tensorboard_cb]
    )

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures={
            'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
            )
        }
    )
