"""TFX pipeline untuk klasifikasi personality menggunakan TFX components."""

import os
from pprint import PrettyPrinter

import tensorflow as tf
from tensorflow.keras import layers

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma
from IPython.display import display

# Konfigurasi pipeline
PIPELINE_NAME = "personality-pipeline"
PIPELINE_ROOT = os.path.join("pipelines", PIPELINE_NAME)
METADATA_PATH = os.path.join("metadata", PIPELINE_NAME, "metadata.db")
SERVING_MODEL_DIR = os.path.join("serving_model", PIPELINE_NAME)
DATA_ROOT = "data"

interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)


def init_components(args=None):
    """Inisialisasi seluruh komponen dalam pipeline TFX."""
    if args is None:
        args = {
            "data_dir": DATA_ROOT,
            "trainer_module": "modules/personality_trainer.py",
            "tuner_module": "modules/personality_tuner.py",
            "transform_module": "modules/personality_transform.py",
            "train_steps": 1000,
            "eval_steps": 500,
            "serving_model_dir": SERVING_MODEL_DIR
        }

    components = {}
    component_list = []

    # ExampleGen
    example_gen = CsvExampleGen(
        input_base=DATA_ROOT,
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=7),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=3)
            ])
        )
    )
    interactive_context.run(example_gen)
    components["example_gen"] = example_gen
    component_list.append(example_gen)

    # Cetak sampel data
    train_uri = os.path.join(
        example_gen.outputs['examples'].get()[0].uri, 'Split-train'
    )
    tfrecord_files = [
        os.path.join(train_uri, f) for f in os.listdir(train_uri)
    ]
    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')

    pp = PrettyPrinter()
    for record in dataset.take(2):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        pp.pprint(example)

    # StatisticsGen
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    interactive_context.run(statistics_gen)
    interactive_context.show(statistics_gen.outputs["statistics"])
    components["statistics_gen"] = statistics_gen
    component_list.append(statistics_gen)

    # SchemaGen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    interactive_context.run(schema_gen)
    interactive_context.show(schema_gen.outputs["schema"])
    components["schema_gen"] = schema_gen
    component_list.append(schema_gen)

    # ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )
    interactive_context.run(example_validator)
    interactive_context.show(example_validator.outputs["anomalies"])
    components["example_validator"] = example_validator
    component_list.append(example_validator)

    # Transform
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(args["transform_module"])
    )
    interactive_context.run(transform)
    components["transform"] = transform
    component_list.append(transform)

    # Tuner
    tuner = Tuner(
        module_file=os.path.abspath(args["tuner_module"]),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=args["train_steps"]),
        eval_args=trainer_pb2.EvalArgs(num_steps=args["eval_steps"])
    )
    interactive_context.run(tuner)
    components["tuner"] = tuner
    component_list.append(tuner)

    # Trainer
    trainer = Trainer(
        module_file=os.path.abspath(args["trainer_module"]),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=args["train_steps"]),
        eval_args=trainer_pb2.EvalArgs(num_steps=args["eval_steps"]),
        hyperparameters=tuner.outputs['best_hyperparameters']
    )
    interactive_context.run(trainer)
    components["trainer"] = trainer
    component_list.append(trainer)

    # Evaluator
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Personality_xf')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[tfma.MetricConfig(class_name='BinaryAccuracy')],
                thresholds={
                    'binary_accuracy': tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.6}
                        )
                    )
                }
            )
        ]
    )

    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config
    )
    interactive_context.run(evaluator)
    components["evaluator"] = evaluator
    component_list.append(evaluator)

    # Tampilkan hasil evaluasi
    eval_result_uri = evaluator.outputs['evaluation'].get()[0].uri
    tfma_result = tfma.load_eval_result(eval_result_uri)
    display(tfma.view.render_slicing_metrics(tfma_result))

    try:
        display(
            tfma.addons.fairness.view.widget_view.render_fairness_indicator(tfma_result)
        )
    except Exception as err:
        print("Fairness indicator gagal ditampilkan:", err)

    # Resolver
    try:
        model_resolver = Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing)
        ).with_id('latest_blessed_model_resolver')

        interactive_context.run(model_resolver)
        components["model_resolver"] = model_resolver
        component_list.append(model_resolver)

    except Exception as err:
        print(f"Model resolver tidak dapat dijalankan: {err}")
        print("Ini normal untuk pipeline pertama kali dijalankan.")

    # Pusher
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs.get('blessing', None),
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=os.path.abspath(
                    'serving_model_dir/personality_model'
                )
            )
        )
    )
    interactive_context.run(pusher)
    components["pusher"] = pusher
    component_list.append(pusher)

    return {
        'components_dict': components,
        'components_list': component_list
    }


if __name__ == "__main__":
    # Jalankan seluruh pipeline
    COMPONENTS = init_components()
