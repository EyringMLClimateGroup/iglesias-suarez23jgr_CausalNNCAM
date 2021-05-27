from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from .cbrain.learning_rate_schedule import LRUpdate
from .cbrain.save_weights import save_norm
from .data_generator import build_train_generator, build_valid_generator
from datetime import datetime


def train_all_models(model_descriptions, setup):
    """ Train and save all the models """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for model_description in model_descriptions:
        train_model(model_description, setup, timestamp)
        save_model(model_description, setup)


def train_model(
    model_description, setup, timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")
):
    """ Train a model """
    print(f"Training {model_description}")

    input_vars_dict = model_description.input_vars_dict
    output_vars_dict = model_description.output_vars_dict

    with build_train_generator(
        input_vars_dict, output_vars_dict, setup
    ) as train_gen, build_valid_generator(
        input_vars_dict, output_vars_dict, setup
    ) as valid_gen:
        lrs = LearningRateScheduler(
            LRUpdate(init_lr=setup.init_lr, step=setup.step_lr, divide=setup.divide_lr)
        )
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=Path(
                model_description.get_path(setup.tensorboard_folder),
                "{timestamp}-{filename}".format(
                    timestamp=timestamp, filename=model_description.get_filename()
                ),
            ),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=setup.train_patience)
        model_description.fit_model(
            x=train_gen,
            validation_data=valid_gen,
            epochs=setup.epochs,
            callbacks=[lrs, tensorboard, early_stop],
            verbose=setup.train_verbose,
        )


def save_model(model_description, setup):
    """ Save all model information necessary for CAM """
    model_description.save_model(setup.nn_output_path)
    # Saving norm after saving the model avoids having to create
    # the folder ourserlves
    save_norm(
        input_transform=train_gen.input_transform,
        output_transform=train_gen.output_transform,
        save_dir=str(model_description.get_path(setup.nn_output_path)),
        filename=model_description.get_filename(),
    )
