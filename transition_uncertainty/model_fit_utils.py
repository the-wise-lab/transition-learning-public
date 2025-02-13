import os
import time
from datetime import datetime
from functools import partialmethod
from typing import Dict, List, Tuple, Union

import bayesflow as bf
import dill
import keras
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from transition_uncertainty.beta_models import (
    simulate_leaky_beta_transition_learner,
    simulate_rescorla_wagner_transition_learner,
)
from transition_uncertainty.modelling_utils import get_simulation_params


def print_timestamped_message(message: str) -> None:
    """
    Print the provided message prefixed with the current timestamp.

    Args:
        message (str): The message to print.

    Returns:
        None
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def train_model_with_params(
    model_type: str,
    N_SUBS: int,
    params_dict: Dict[str, np.ndarray],
    true_params_subset: np.ndarray,
    available_side: np.ndarray,
    common_params: List[Union[float, np.ndarray]],
    n_trials: int = None,
    output_path: str = None,
) -> None:
    """
    Train and save a model with specified parameters.

    Args:
        model_type (str): Type of the model to simulate and save.
        N_SUBS (int): Number of subjects.
        params_dict (dict): Dictionary containing the parameters to be
            used for simulation.
        true_params_subset (np.ndarray): Subset of true parameters to use.
        available_side (np.ndarray): Array of shape (n_subs, n_trials)
            containing the available side on each trial for each subject.
        common_params (list): Parameters common across all model types. These
            are entered into the simulation call after the other values.
            Generally, these should correspond to: [starting_value_estimate,
            starting_transition_prob_estimate, second_stage_states_all,
            reward_probs_all, rewards_all, available_side_all]
        n_trials (int):Number of trials to simulate. If None,
            all trials are used. output_path
        (str): Path to save the model. If None, the model is saved in the
        default location (results/behavioural_models/).

    This function performs the following steps:
    - Computes the simulation parameters for the specified
        model type by calling `get_simulation_params`.
    - Generates a path for saving the model based on the model
        type and number of subjects.
    - Calls `train_and_save_model` with the assembled parameters
        to train the model and save the results.
    """
    # Path where the model will be saved
    if output_path is None:
        if n_trials is None:
            model_path = (
                f"results/behavioural_models/npe_{model_type}_{N_SUBS}.pkl"
            )
        else:
            model_path = f"results/behavioural_models/npe_{model_type}_{N_SUBS}_{n_trials}.pkl"
    else:
        if n_trials is None:
            model_path = f"{output_path}/npe_{model_type}_{N_SUBS}.pkl"
        else:
            model_path = (
                f"{output_path}/npe_{model_type}_{N_SUBS}_{n_trials}.pkl"
            )

    # Get the simulation parameters for the specified model type
    simulate_params = get_simulation_params(
        model_type, params_dict, params_dict["W"], common_params
    )

    # Call the function to train and save the model
    train_and_save_model(
        simulate_params,
        true_params_subset,
        available_side,
        model_type,
        model_path,
        n_trials,
    )


class FlowMatchingSteps(bf.networks.FlowMatching):
    """This enables us to set the number of steps in network, since
    the default is 100 which is not necessary and makes things very slow
    """

    _forward = partialmethod(bf.networks.FlowMatching._forward, steps=5)
    _inverse = partialmethod(bf.networks.FlowMatching._inverse, steps=5)


class BayesFlowModel:

    def __init__(
        self,
        train_proportion: float = 0.80,
        batch_size: int = 256,
        epochs: int = 32,
    ):
        self.adapter = (
            bf.adapters.Adapter()
            # convert any non-arrays to numpy arrays
            .to_array()
            # convert from numpy's default float64 to deep learning friendly float32
            .convert_dtype("float64", "float32")
            # Constrain to be within the range [0, 1]
            .constrain("inference_variables", lower=0, upper=1)
            # standardize all variables to zero mean and unit variance aside from behaviour
            .standardize(exclude=["inference_conditions", "summary_variables"])
        )

        self.train_proportion = train_proportion
        self.batch_size = batch_size
        self.epochs = epochs

        # Set up the inference network
        inference_network = FlowMatchingSteps(
            subnet="mlp",
            subnet_kwargs={
                "widths": (256, 256),
                "dropout": 0.1,
            },
        )

        # Set up the approximator
        self.approximator = bf.ContinuousApproximator(
            inference_network=inference_network,
            adapter=self.adapter,
        )

    def fit(self, behaviour: np.ndarray, params: np.ndarray):

        # Get number of training batches
        num_training_batches = int(
            self.train_proportion * len(params) / self.batch_size
        )

        # Get total number of steps
        total_steps = num_training_batches * self.epochs

        # Cosine learning rate schedule
        scheduled_lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=5e-3,
            decay_steps=total_steps,
            alpha=1e-8,
        )

        optimizer = keras.optimizers.AdamW(learning_rate=scheduled_lr)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # Compile
        self.approximator.compile(optimizer=optimizer)

        # Set up training and validation data
        training_samples = {
            "inference_variables": params[
                : num_training_batches * self.batch_size
            ],
            "inference_conditions": np.array(
                behaviour[: num_training_batches * self.batch_size]
            )
            .astype("float32")
            .squeeze(),
        }

        validation_samples = {
            "inference_variables": params[
                num_training_batches * self.batch_size :
            ],
            "inference_conditions": np.array(
                behaviour[num_training_batches * self.batch_size :]
            )
            .astype("float32")
            .squeeze(),
        }

        # Create OfflineDatasets
        training_dataset = bf.datasets.OfflineDataset(
            data=training_samples,
            batch_size=self.batch_size,
            adapter=self.adapter,
        )

        validation_dataset = bf.datasets.OfflineDataset(
            data=validation_samples,
            batch_size=self.batch_size,
            adapter=self.adapter,
        )

        # Fit
        self.history = self.approximator.fit(
            epochs=self.epochs,
            dataset=training_dataset,
            validation_data=validation_dataset,
            callbacks=[early_stopping],
        )

    def sample(
        self,
        behaviour: np.ndarray,
        n_samples: int = 1000,
        progress_bar: bool = False,
    ):

        # Set up inference conditions
        conditions = {}
        conditions["inference_conditions"] = (
            np.array(behaviour).astype("float32").squeeze()
        )

        # Sample parameters]
        estimated_params = self.approximator.sample(
            conditions=conditions,
            batch_size=behaviour.shape[0],
            num_samples=n_samples,
        )["inference_variables"]

        return estimated_params.transpose(1, 0, 2)


def train_and_save_model(
    simulate_params: Tuple[np.ndarray, ...],
    true_params: np.ndarray,
    available_side: np.ndarray,
    model_name: str,
    model_path: str,
    n_trials: int = None,
) -> None:
    """
    Train and save a model given simulation and fitting parameters.

    Args:
        simulate_params (tuple): Parameters to be passed to
            `simulate_leaky_beta_transition_learner` function or
            `simulate_rescorla_wagner_transition_learner` function.
        true_params (np.ndarray): True parameters to be passed to the `fit`
            method of `NPEModel` instance.
        available_side (np.ndarray): Array of shape (n_subs, n_trials)
            containing the available side on each trial for each subject.
        model_name (str): A string to display in training log, indicating which
            model is being trained.
        model_path (str): File path (including name) to save the trained model.
        n_trials (int): Number of trials to simulate. If None, all trials are
        used.

    Returns:
        None
    """
    # Notify the user about the model being trained.
    print_timestamped_message(f"Training {model_name} model...")

    # Notify the user that the data simulation is starting.
    print_timestamped_message("Simulating data...")

    # Call the simulation function
    if "rw" in model_name:
        _, choices, _, _, _, _, _, _, _ = (
            simulate_rescorla_wagner_transition_learner(*simulate_params)
        )
    else:
        _, choices, _, _, _, _, _, _, _ = (
            simulate_leaky_beta_transition_learner(*simulate_params)
        )

    # Notify the user that the simulation is complete.
    print_timestamped_message("Simulation completed.")

    # Filter the 'choices' array to remove certain trials.
    print_timestamped_message(
        "Filtering choices array based on available_side..."
    )
    choices = choices[:, :, available_side[0, :] == -1]

    # Cap the number of trials based on the n_trials parameter.
    if n_trials is not None:
        choices = choices[:, :, :n_trials]

    # Initialize an instance of NPEModel, specifying the format of the choices.
    print_timestamped_message("Initializing NPEModel instance...")
    # npe_model = NPEModel(choice_format="numerical")
    npe_model = BayesFlowModel()

    # Notify the user that the model fitting is starting.
    print_timestamped_message("Starting model fitting...")

    # Record the start time of the fitting.
    start_time = time.time()

    # Fit the model using 'choices' and 'true_params'.
    npe_model.fit(choices, true_params)

    # Record the end time of the fitting.
    end_time = time.time()

    # Calculate and print the time taken for fitting.
    fit_time = end_time - start_time
    print_timestamped_message(f"Fitting completed in {fit_time:.2f} seconds.")

    # Check if the output directory exists, and create it if it doesn't.
    print_timestamped_message("Checking output directory...")
    if not os.path.exists(os.path.dirname(model_path)):
        print_timestamped_message(
            "Output directory does not exist. Creating it..."
        )
        os.makedirs(os.path.dirname(model_path))

    # Open a file in write-binary mode and save the trained model using dill.
    print_timestamped_message("Saving the model...")
    with open(model_path, "wb") as f:
        dill.dump(npe_model, f)

    # Notify the user that the model has been saved.
    print_timestamped_message(f"Model saved successfully at {model_path}.")
