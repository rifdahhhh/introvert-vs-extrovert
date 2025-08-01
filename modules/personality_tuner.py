"""Tuning module for personality classification using KerasTuner."""

from collections import namedtuple
import tensorflow as tf
import tensorflow_transform as tft
import kerastuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

TunerFnResult = namedtuple('TunerFnResult', ['tuner', 'fit_kwargs'])

LABEL_KEY = "Personality"

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

def input_fn(file_pattern, tf_transform_output, num_epochs=1, batch_size=32):
    """Create input dataset for tuner."""
    if isinstance(file_pattern, list):
        all_files = []
        for pattern in file_pattern:
            all_files.extend(tf.io.gfile.glob(pattern))
        file_pattern = all_files

    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
        shuffle=True,
        shuffle_buffer_size=1000
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

def build_model(hp):
    """Build and compile Keras model with hyperparameters."""
    inputs = {}
    encoded_inputs = []

    for key in CATEGORICAL_FEATURE_KEYS:
        input_key = transformed_name(key)
        inputs[input_key] = tf.keras.Input(shape=(1,), name=input_key, dtype=tf.int64)
        embedding = layers.Embedding(input_dim=10, output_dim=4)(inputs[input_key])
        flat = layers.Flatten()(embedding)
        encoded_inputs.append(flat)

    for key in NUMERIC_FEATURE_KEYS:
        input_key = transformed_name(key)
        inputs[input_key] = tf.keras.Input(shape=(1,), name=input_key, dtype=tf.float32)
        encoded_inputs.append(inputs[input_key])

    x = layers.Concatenate()(encoded_inputs)
    x = layers.Dense(hp.Int("units_1", 32, 128, step=16), activation='relu')(x)
    x = layers.Dense(hp.Int("units_2", 16, 64, step=16), activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [0.001, 0.01, 0.1])
        ),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """TFX entry point for tuning."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output,
                         num_epochs=None, batch_size=32)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output,
                        num_epochs=None, batch_size=32)

    tuner = kerastuner.RandomSearch(
        build_model,
        max_trials=5,
        objective='val_binary_accuracy',
        directory=fn_args.working_dir,
        project_name='personality_tuning',
        max_consecutive_failed_trials=5
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": 50,
            "validation_steps": 25,
            "epochs": 5,
            "verbose": 1
        }
    )
