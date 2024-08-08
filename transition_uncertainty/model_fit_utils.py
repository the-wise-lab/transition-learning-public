import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Union

import dill
import numpy as np
from simulation_based_inference.npe import NPEModel

from transition_uncertainty.beta_models import (
    simulate_leaky_beta_transition_learner,
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
            Generally, these should correspond to: [temperature,
            starting_value_estimate, starting_transition_prob_estimate,
            second_stage_states_all, reward_probs_all, rewards_all,
            available_side_all] 
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
            `simulate_leaky_beta_transition_learner` function.
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

    # Call the simulation function and unpack the return values.
    _, choices, _, _, _, _, _, _, _ = simulate_leaky_beta_transition_learner(
        *simulate_params
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
    npe_model = NPEModel(choice_format="numerical")

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
