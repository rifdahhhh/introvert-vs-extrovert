
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os
from kerastuner import HyperParameters

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
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs=10, batch_size=64):
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
    inputs = {}
    encoded_inputs = []

    # Handle kategorikal
    for key in CATEGORICAL_FEATURE_KEYS:
        feat_key = transformed_name(key)
        inputs[feat_key] = tf.keras.Input(shape=(1,), name=feat_key, dtype=tf.int64)
        embed = tf.keras.layers.Embedding(input_dim=10, output_dim=4)(inputs[feat_key])
        flat = tf.keras.layers.Flatten()(embed)
        encoded_inputs.append(flat)

    # Handle numerik
    for key in NUMERIC_FEATURE_KEYS:
        feat_key = transformed_name(key)
        inputs[feat_key] = tf.keras.Input(shape=(1,), name=feat_key, dtype=tf.float32)
        encoded_inputs.append(inputs[feat_key])

    # Gabungkan semua input
    x = tf.keras.layers.Concatenate()(encoded_inputs)

    # Gunakan hasil tuning jika ada
    if hp:
        units_1 = hp.Int("units_1", 32, 128, step=16)
        units_2 = hp.Int("units_2", 16, 128, step=16)
        learning_rate = hp.Choice("learning_rate", [0.001, 0.01, 0.1])
    else:
        units_1 = 64
        units_2 = 32
        learning_rate = 0.001

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
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    # Gunakan hasil tuning
    if fn_args.hyperparameters:
        hparams = HyperParameters.from_config(fn_args.hyperparameters)
        model = model_builder(hp=hparams)
    else:
        model = model_builder()

    # Callback untuk TensorBoard
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs'))

    # Training
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=100,
        validation_steps=50,
        epochs=5,
        callbacks=[tensorboard_cb]
    )

    # Simpan model untuk serving
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures={
            'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
            )
        }
    )
