import os
import re
import warnings
from typing import Any, Dict, List, Tuple, Union

import arviz as az
import dill
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .beta_models import simulate_leaky_beta_transition_learner
from .utils import tqdm_joblib

os.environ["OUTDATED_IGNORE"] = "1"


def transform_to_bounded(x: float, lower: float, upper: float) -> float:
    """
    Transforms a value from the unit interval to a bounded interval.

    Args:
        x (float): The value to transform, in the unit interval [0, 1].
        lower (float): The lower bound of the target interval.
        upper (float): The upper bound of the target interval.

    Returns:
        float: The transformed value in the target interval [lower, upper].
    """
    return lower + (upper - lower) * x


def transform_from_bounded(x: float, lower: float, upper: float) -> float:
    """
    Transforms a value from a bounded interval to the unit interval.

    Args:
        x (float): The value to transform, in the target interval [lower,
        upper].
        lower (float): The lower bound of the interval.
        upper (float): The upper bound of the interval.

    Returns:
        float: The transformed value in the unit interval [0, 1].
    """
    return (x - lower) / (upper - lower)


def load_task_spec(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the task specification data from the given directory.

    Args:
        data_dir (str): The directory containing the task specification data.

    Returns:
        A tuple containing the following elements:
            - second_stage_states (np.ndarray): The second stage states.
            - second_stage_state_probs (np.ndarray): The probabilities of the
              second stage states.
            - rewards (np.ndarray): The rewards.
            - reward_probs (np.ndarray): The probabilities of the rewards.
            - available_side (np.ndarray): The available side (on confidence
              trials).
    """
    second_stage_states = np.load(
        os.path.join(data_dir, "second_stage_states.npy")
    )
    second_stage_state_probs = np.load(
        os.path.join(data_dir, "second_stage_state_probs.npy")
    )
    rewards = np.load(os.path.join(data_dir, "rewards.npy"))
    reward_probs = np.load(os.path.join(data_dir, "reward_probs.npy"))
    available_side = np.load(os.path.join(data_dir, "available_side.npy"))
    return (
        second_stage_states,
        second_stage_state_probs,
        rewards,
        reward_probs,
        available_side,
    )


def repeat_for_all_subjects(
    array: np.ndarray, n_subs: int, extra_axis: int = None
) -> np.ndarray:
    """
    Repeat the input array for all subjects along a new leading axis.

    Args:
        array (np.array): The array to be repeated.
        n_subs (int): The number of subjects for repetition.
        extra_axis (int, optional): Additional axis to repeat
            the array. Default is None.

    Returns: np.array: An array repeated for all subjects.
    """
    if extra_axis is not None:
        repeated_array = np.repeat(array[None, ..., None], n_subs, axis=0)
    else:
        repeated_array = np.repeat(array[None, ...], n_subs, axis=0)

    return repeated_array


def generate_simulation_parameters(
    n_subs: int, task_spec_path: str, random_seed: int = 42
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate simulation parameters and load task specification.

    Args:
        n_subs (int): Number of subjects for the simulation.
        task_spec_path (str): Path to the task specification data file.
        random_seed (int, optional): Seed for random number generator.
            Default is 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]:
            - second_stage_states_all (np.ndarray): Replicated second stage
              states for all subjects.
            - rewards_all (np.ndarray): Replicated rewards for all subjects.
            - reward_probs_all (np.ndarray): Replicated reward probabilities
              for all subjects.
            - available_side_all (np.ndarray): Replicated available side
              information for all subjects.
            - starting_value_estimate (np.ndarray): Starting values for value
              estimates, shaped (n_subs,).
            - starting_transition_prob_estimate (np.ndarray): Starting values
              for transition probability estimates, shaped (n_subs,).
            - true_params_array (np.ndarray): True parameter values, stacked
              for training, shaped (n_subs, number_of_params) and
            ordered as [tau_value, tau_prob, decay_value, decay_prob, W,
            temperature]. Values are transformed to be in the range [0, 1].

    """
    # Load task specification
    (
        second_stage_states,
        second_stage_state_probs,
        rewards,
        reward_probs,
        available_side,
    ) = load_task_spec(task_spec_path)

    # Replicate task variables for each subject
    second_stage_states_all = repeat_for_all_subjects(
        second_stage_states, n_subs
    )
    rewards_all = repeat_for_all_subjects(rewards, n_subs)
    reward_probs_all = repeat_for_all_subjects(reward_probs, n_subs)
    available_side_all = repeat_for_all_subjects(
        available_side, n_subs, extra_axis=-1
    )

    # Set up a random number generator with a fixed seed to ensure
    # reproducibility
    rng = np.random.RandomState(random_seed)

    # Generate simulation parameters using uniform distributions
    tau_value = rng.uniform(0, 1, size=n_subs)
    tau_prob = rng.uniform(0, 1, size=n_subs)
    decay_value = rng.uniform(0, 1, size=n_subs)
    decay_prob = rng.uniform(0, 1, size=n_subs)
    W = rng.uniform(0.1, 0.9, size=n_subs)
    temperature = rng.uniform(0.01, 0.2, size=n_subs)

    # Put into a dictionary to return
    params_dict = {
        "tau_value": tau_value,
        "tau_prob": tau_prob,
        "decay_value": decay_value,
        "decay_prob": decay_prob,
        "W": W,
        "temperature": temperature,
    }

    # Define fixed starting values for simulations
    starting_value_estimate = np.ones(n_subs)
    starting_transition_prob_estimate = np.ones(n_subs)

    # Stack true parameter values as targets for training
    true_params_array = np.vstack(
        [
            tau_value,
            tau_prob,
            decay_value,
            decay_prob,
            transform_from_bounded(W, 0.1, 0.9),
            transform_from_bounded(temperature, 0.01, 0.2),
        ]
    ).T

    return (
        second_stage_states_all,
        rewards_all,
        reward_probs_all,
        available_side_all,
        available_side,
        starting_value_estimate,
        starting_transition_prob_estimate,
        params_dict,
        true_params_array,
    )


def get_simulation_params(
    model_type: str,
    params_dict: Dict[str, Union[int, float, np.ndarray]],
    W_like: np.ndarray,
    common_params: List[Union[float, np.ndarray]],
) -> List[Union[float, np.ndarray]]:
    """
    Get simulation parameters based on the model type.

    Args:
        model_type (str): Type of the model to simulate.
        params_dict (dict): Dictionary containing parameter values.
        W_like (np.ndarray): A numpy array to derive shapes for
            creating arrays of ones and zeros.
        common_params (list): Parameters common across all model types. These
            are entered into the simulation call after the other values.
            Generally, these should correspond to: [temperature,
            starting_value_estimate, starting_transition_prob_estimate,
            second_stage_states_all, reward_probs_all, rewards_all,
            available_side_all]

    Returns:
        list: Simulation parameters assembled based on the model type.

    This function takes a model type as input and uses it to decide which
    parameters to prepare and return for the simulation. The returned list of
    parameters will be used to simulate the specified model type in the later
    stages of the program.
    """

    # Only using model-free value learning
    if model_type == "mf_only":
        return [
            params_dict["tau_value"],
            params_dict["tau_value"],
            np.ones_like(W_like) * 0,
            params_dict["decay_value"],
            np.ones_like(W_like) * 0,
            np.ones_like(W_like) * 0,
        ] + common_params

    # Only using model-based decision-making based on learnt transition
    # probabilities
    elif model_type == "mb_only":
        return [
            np.ones_like(W_like) * 0,
            np.ones_like(W_like) * 0,
            params_dict["tau_prob"],
            np.ones_like(W_like) * 0,
            params_dict["decay_prob"],
            np.ones_like(W_like) * 1,
        ] + common_params

    # Combining MF and MB according to weighting parameter W
    elif model_type == "weighting":
        return [
            params_dict["tau_value"],
            params_dict["tau_value"],
            params_dict["tau_prob"],
            params_dict["decay_value"],
            params_dict["decay_prob"],
            params_dict["W"],
        ] + common_params

    # Taking the simple average of MF and MB (W = 0.5)
    elif model_type == "weighting_fixed":
        return [
            params_dict["tau_value"],
            params_dict["tau_value"],
            params_dict["tau_prob"],
            params_dict["decay_value"],
            params_dict["decay_prob"],
            np.ones_like(W_like) * 0.5,
        ] + common_params


def calculate_waic(
    sampled_params: np.ndarray,
    second_stage_states: np.ndarray,
    rewards: np.ndarray,
    reward_probs: np.ndarray,
    available_side: np.ndarray,
    test_choices: np.ndarray,
    pointwise: bool = False,
) -> Tuple[az.InferenceData, az.ELPDData]:
    """
    Calculates the Widely Applicable Information Criterion (WAIC) for a given
    set of parameter samples and observed choices.

    This works by taking the estimated parameters and simulating data from the
    candidate model using these parameters. This is then used to get the logp
    of the data, which is used to calculate the WAIC.

    Args:
        sampled_params (np.ndarray): Array of shape (n_samples, n_subjects,
            n_params) containing parameter samples.
        second_stage_states (np.ndarray): Array of shape
            (n_blocks, n_trials, n_options) containing the second stage states.
        rewards (np.ndarray): Array of shape (n_blocks, n_trials, n_options)
            containing the rewards.
        reward_probs (np.ndarray): Array of shape
            (n_blocks, n_trials, n_options) containing  the reward
            probabilities.
        available_side (np.ndarray): Array of shape (n_blocks, n_trials)
            containing the available options.
        test_choices (np.ndarray): Array of shape
        (n_subjects, n_blocks, n_trials) containing the observed choices
            for each observation. Should be given with confidence trials
            included.
        pointwise (bool, optional): Whether to  return pointwise WAIC.
            Defaults to False.

    Returns:
        Tuple[az.InferenceData, az.ELPDData]: A tuple containing the
        InferenceData object and the WAIC.
    """

    # Check shapes
    assert sampled_params.ndim == 3, "sampled_params must have 3 dimensions"
    assert (
        second_stage_states.ndim == 3
    ), "second_stage_states must have 3 dimensions"
    assert rewards.ndim == 3, "rewards must have 3 dimensions"
    assert reward_probs.ndim == 3, "reward_probs must have 3 dimensions"
    assert available_side.ndim == 2, "available_side must have 2 dimensions"
    assert test_choices.ndim == 3, "test_choices must have 3 dimensions"
    assert (
        second_stage_states.shape[0]
        == rewards.shape[0]
        == reward_probs.shape[0]
        == available_side.shape[0]
    ), "second_stage_states, rewards, reward_probs, and available_side must "
    "have the same number of blocks"
    assert (
        second_stage_states.shape[1]
        == rewards.shape[1]
        == reward_probs.shape[1]
        == available_side.shape[1]
    ), "second_stage_states, rewards, reward_probs, available_side must "
    "have the same number of trials"

    # Get dimensions
    n_parameters = sampled_params.shape[-1]
    n_samples = sampled_params.shape[0]
    n_subjects = sampled_params.shape[1]

    # Extract parameter samples
    # Samples become subjects nested within samples (e.g., sub1_samp1,
    # sub2_samp1...)
    sampled_params_reshaped = sampled_params.reshape(-1, n_parameters)

    # Get combined observations (samples x observations)
    n_obs = sampled_params_reshaped.shape[0]

    # Fixed starting values
    starting_value_estimate = np.ones(n_obs)
    starting_transition_prob_estimate = np.ones(n_obs)

    # Replicate task variables for each subject
    second_stage_states_all = repeat_for_all_subjects(
        second_stage_states, n_obs
    )
    rewards_all = repeat_for_all_subjects(rewards, n_obs)
    reward_probs_all = repeat_for_all_subjects(reward_probs, n_obs)
    available_side_all = repeat_for_all_subjects(
        available_side, n_obs, extra_axis=-1
    )
    # Tile test choices for each sample
    test_choices_all = np.tile(test_choices, (n_samples, 1, 1))

    # Simulate data using parameter samples
    # This is necessary to get the logp of the data
    choice_p, _, _, _, _, _, _, _, _ = simulate_leaky_beta_transition_learner(
        sampled_params_reshaped[:, 0],
        sampled_params_reshaped[:, 0],
        sampled_params_reshaped[:, 1],
        sampled_params_reshaped[:, 2],
        sampled_params_reshaped[:, 3],
        sampled_params_reshaped[:, 4],
        sampled_params_reshaped[:, 5],
        starting_value_estimate,
        starting_transition_prob_estimate,
        second_stage_states_all,
        reward_probs_all,
        rewards_all,
        available_side_all,
        observed_choices=test_choices_all,
    )

    # remove any parameters where values are the same for all observations
    sampled_params = sampled_params[
        ..., ~(sampled_params.reshape(-1, 6).std(axis=0) == 0)
    ]

    # Reshape choice probability to be samples x ...
    # choice_p_reshaped = choice_p.reshape(n_samples, n_subjects, n_trials, 2)
    # Remove confidence trials
    choice_p = choice_p[..., available_side[0, :] == -1, :]
    test_choices_all = test_choices_all[..., available_side[0, :] == -1]

    # Get pointwise log likelihood
    log_likelihood = dist.Bernoulli(probs=choice_p[..., 1]).log_prob(
        test_choices_all
    )

    # Reshape so that the log likelihood shape is (1, n_samples, n_subjects,
    # n_trials)
    log_likelihood = log_likelihood.reshape(
        1, n_samples, n_subjects, choice_p.shape[-2]
    )

    # convert to inference data
    dataset = az.convert_to_inference_data(
        sampled_params.reshape(
            n_samples, n_subjects, sampled_params.shape[-1]
        )[
            None,
        ]
    )

    # add likelihood to dataset
    dataset.add_groups({"log_likelihood": {"log_likelihood": log_likelihood}})

    # Get WAIC
    with warnings.catch_warnings():  # suppress variance warning
        warnings.simplefilter("ignore")
        waic = az.waic(dataset, pointwise=pointwise)

    return dataset, waic


def simulate_from_mean_params(
    mean_params: np.ndarray,
    second_stage_states: np.ndarray,
    rewards: np.ndarray,
    reward_probs: np.ndarray,
    available_side: np.ndarray,
) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """
    Simulates data given mean parameter estimates.

    Args:
        mean_params (np.ndarray): Array of shape (n_subjects, n_params)
            containing parameter estimates.
        second_stage_states (np.ndarray): Array of shape
            (n_blocks, n_trials, n_options) containing the second stage
            states.
        rewards (np.ndarray): Array of shape (n_blocks, n_trials,
            n_options) containing the rewards.
        reward_probs (np.ndarray): Array of shape (n_blocks,
            n_trials, n_options) containing the reward probabilities.
        available_side (np.ndarray): Array of shape (n_blocks,
            n_trials) containing the available options.

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array]: Tuple of arrays
        containing the choice probability, transition probabilities,
        and distributions representing combined value.
    """

    # Check shapes
    assert mean_params.ndim == 2, "mean_params must have 2 dimensions"
    assert (
        second_stage_states.ndim == 3
    ), "second_stage_states must have 3 dimensions"
    assert rewards.ndim == 3, "rewards must have 3 dimensions"
    assert reward_probs.ndim == 3, "reward_probs must have 3 dimensions"
    assert available_side.ndim == 2, "available_side must have 2 dimensions"
    assert (
        second_stage_states.shape[0]
        == rewards.shape[0]
        == reward_probs.shape[0]
        == available_side.shape[0]
    ), "second_stage_states, rewards, reward_probs, and available_side "
    "must have the same number of blocks"
    assert (
        second_stage_states.shape[1]
        == rewards.shape[1]
        == reward_probs.shape[1]
        == available_side.shape[1]
    ), "second_stage_states, rewards, reward_probs, available_side must "
    "have the same number of trials"

    # Get dimensions
    n_subjects = mean_params.shape[0]

    # Fixed starting values
    starting_value_estimate = np.ones(n_subjects)
    starting_transition_prob_estimate = np.ones(n_subjects)

    # Task parameters
    # repeat second stage states for each observation
    second_stage_states_all = np.repeat(
        second_stage_states[None, :], n_subjects, axis=0
    )
    # repeat rewards for each observation
    rewards_all = np.repeat(rewards[None, ...], n_subjects, axis=0)
    # repeat reward probs for each observation
    reward_probs_all = np.repeat(reward_probs[None, ...], n_subjects, axis=0)
    # repeat confidence available observation
    available_side_all = np.repeat(
        available_side[None, ..., None], n_subjects, axis=0
    )

    # Simulate data using parameter samples
    (
        choice_p,
        _,
        _,
        transition_estimates,
        _,
        _,
        combined_dists,
        _,
        _,
    ) = simulate_leaky_beta_transition_learner(
        mean_params[:, 0],
        mean_params[:, 0],
        mean_params[:, 1],
        mean_params[:, 2],
        mean_params[:, 3],
        transform_to_bounded(mean_params[:, 4], 0.1, 0.9),
        transform_to_bounded(mean_params[:, 5], 0.01, 0.2),
        starting_value_estimate,
        starting_transition_prob_estimate,
        second_stage_states_all,
        reward_probs_all,
        rewards_all,
        available_side_all,
    )

    return choice_p, transition_estimates, combined_dists


def simulate_model_with_params(
    model_type: str,
    params_dict: Dict[str, np.ndarray],
    available_side: np.ndarray,
    common_params: List[Union[float, np.ndarray]],
    remove_confidence_trials: bool = False,
    random_seed: int = 42,
) -> None:
    """
    Train and save a model with specified parameters.

    Args:
        model_type (str): Type of the model to simulate and save.
        param_dict (dict): Dictionary containing the parameters
            to be used for simulation.
        available_side (np.ndarray): Array of shape (n_subs, n_trials)
            containing the available side on each trial for each subject.
        common_params (list): Parameters common across all model types. These
            are entered into the simulation call after the other values.
            Generally, these should correspond to: [temperature,
            starting_value_estimate, starting_transition_prob_estimate,
            second_stage_states_all, reward_probs_all, rewards_all,
            available_side_all]
        remove_confidence_trials (bool, optional): Whether to remove confidence
            trials from the simulated data. Defaults to False.
        random_seed (int, optional): Seed for random number generator.
            Default is 42.

    This function performs the following steps: - Computes the simulation
    parameters for the specified model type by calling `get_simulation_params`.
    - Simulates data using the parameters.
    """

    # Get the simulation parameters for the specified model type
    simulate_params = get_simulation_params(
        model_type, params_dict, np.ones_like(params_dict["W"]), common_params
    )

    # Call the simulation function
    _, choices, _, _, _, _, _, _, _ = simulate_leaky_beta_transition_learner(
        *simulate_params, seed=random_seed
    )

    # Optionally remove confidence trials
    if remove_confidence_trials:
        choices = choices[:, :, available_side[0, :] == -1]

    return choices


def find_trained_models(models: list, trained_model_dir: str) -> dict:
    """
    Find and load the model with the most training samples in the directory
    using dill.

    Args:
        models (list): A list containing the names of models.
        trained_model_dir (str): The directory where trained models are saved.

    Returns:
        dict: A dictionary containing the loaded models mapped to model names.

    Raises:
        FileNotFoundError: If one or more of the specified models are not
        found.

    This function navigates through the specified directory, identifies all
    instances of the specified models, chooses the ones with the most training
    samples, loads them using dill, and returns them in a dictionary. Warnings
    are issued if multiple instances of a model exist or if different models
    are trained on different numbers of samples.
    """
    most_trained_models = {}
    max_samples_per_model = {}

    # Regex pattern to extract model name and training samples from filename
    pattern = re.compile(r"npe_(?P<model_name>\w+)_(?P<samples>\d+)")

    for filename in os.listdir(trained_model_dir):
        match = pattern.match(filename)
        if match:
            model_name = match.group("model_name")
            samples = int(match.group("samples"))

            if model_name in models:
                # If model has already been found before
                if model_name in most_trained_models:
                    prev_samples = max_samples_per_model[model_name]
                    prev_filename = most_trained_models[model_name]

                    warnings.warn(
                        f"Multiple instances of {model_name} found. "
                        f"Samples in current: {samples}, "
                        f"in previous: {prev_samples}. "
                        f"Using: {filename if samples > prev_samples else prev_filename}."
                    )

                    # Check and update if the current file has more training
                    # samples
                    if samples > prev_samples:
                        most_trained_models[model_name] = filename
                        max_samples_per_model[model_name] = samples
                else:
                    most_trained_models[model_name] = filename
                    max_samples_per_model[model_name] = samples

    # Check if different models are trained on different numbers of samples
    if len(set(max_samples_per_model.values())) > 1:
        warnings.warn(
            "Different models are trained on different numbers of samples."
        )

    # Check if all models were found
    for model in models:
        if model not in most_trained_models:
            raise FileNotFoundError(
                f"No instances of model {model} found in {trained_model_dir}."
            )

    # Load the models using dill and return
    loaded_models = {}
    for model_name, filename in most_trained_models.items():
        with open(os.path.join(trained_model_dir, filename), "rb") as file:
            loaded_models[model_name] = dill.load(file)

    return loaded_models


def map_sampled_params(
    model: str, params: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """
    Map sampled parameters based on the model type to a specified format or
    template.

    Args:
        model (str): The name of the model.
        params (np.ndarray): A 3D NumPy array of parameters.
        template (np.ndarray): A 3D NumPy array template.

    Returns:
        np.ndarray: The mapped parameter array.
    """
    # Copy the template to prevent modifying the original
    mapped_params = np.copy(template)

    if model == "weighting_fixed":
        mapped_params[..., :4] = params[..., :4]
        mapped_params[..., 5] = params[..., 4]
        mapped_params[..., 4] = 0.5

    elif model == "mb_only":
        mapped_params[..., 1] = params[..., 0]
        mapped_params[..., 3] = params[..., 1]
        mapped_params[..., 5] = params[..., 2]
        mapped_params[..., 4] = 1

    elif model == "mf_only":
        mapped_params[..., 0] = params[..., 0]
        mapped_params[..., 2] = params[..., 1]
        mapped_params[..., 5] = params[..., 2]
        mapped_params[..., 4] = 0

    elif model == "weighting":
        mapped_params = params

    else:
        raise ValueError(f"Unknown model: {model}")

    return mapped_params


def run_combination(
    simulated_model: str,
    simulated_data: np.ndarray,
    estimation_model: str,
    estimation_trained_model: Any,
    iteration: int,
    n_subjects: int,
    n_samples: int,
    available_side: np.ndarray,
    second_stage_states_all: np.ndarray,
    rewards_all: np.ndarray,
    reward_probs_all: np.ndarray,
    sampled_param_template: np.ndarray,
) -> pd.DataFrame:
    """
    Run a single combination of simulation model, estimation model, and
    iteration.

    Args:
        simulated_model (str): Name of the simulated model.
        simulated_data (np.ndarray): Simulated data corresponding
            to the simulated model.
        estimation_model (str): Name of the estimation model.
        estimation_trained_model (Any): Trained estimation model.
        iteration (int): Iteration number.
        n_subjects (int): Number of subjects in the simulated data.
        n_samples (int): Number of samples for parameter estimation.
        available_side (np.ndarray): Array indicating available
            side for each trial.
        second_stage_states_all (np.ndarray): All second stage states.
        rewards_all (np.ndarray): All rewards.
        reward_probs_all (np.ndarray): All reward probabilities.
        sampled_param_template (np.ndarray): Template array for sampled
            parameters.

    Returns:
        pd.DataFrame: DataFrame containing results for the particular
            combination.

    Note:
        Ensure all functions called within have appropriate definitions and
        imports in your script.
    """
    # Sample parameters from the trained estimation model.
    sampled_params = estimation_trained_model.sample(
        simulated_data[
            iteration * n_subjects : (iteration + 1) * n_subjects,
            :,
            available_side[0, :] == -1,
        ],
        n_samples=n_samples,
        progress_bar=False,
    )

    # Map sampled parameters to the template array.
    sampled_params = map_sampled_params(
        estimation_model, sampled_params, sampled_param_template
    )

    # Transform parameters from the range [0, 1] to their original ranges.
    sampled_params[..., 4] = transform_to_bounded(
        sampled_params[..., 4], 0.1, 0.9
    )
    sampled_params[..., 5] = transform_to_bounded(
        sampled_params[..., 5], 0.01, 0.2
    )

    # Calculate WAIC and get dataset
    ds, waic = calculate_waic(
        sampled_params,
        second_stage_states_all[0, ...],
        rewards_all[0, ...],
        reward_probs_all[0, ...],
        available_side,
        simulated_data[iteration * n_subjects : (iteration + 1) * n_subjects],
        pointwise=True,
    )

    # Create a DataFrame with results and return it
    return pd.DataFrame(
        {
            "simulated_model": [simulated_model],
            "estimation_model": [estimation_model],
            "iteration": [iteration],
            "waic": [waic.elpd_waic],
            "waic_se": [waic.se],
            "log_likelihood": [
                ds["log_likelihood"]["log_likelihood"].values.mean()
            ],
        }
    )


def run_model_recovery(
    task_spec_path: str,
    n_iterations: int = 5,
    trained_model_dir: str = None,
    n_samples: int = 500,
    n_subjects: int = 200,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run model recovery with parallel processing across different model
    combinations and iterations.

    Args:
        task_spec_path (str): Path to the task specification file.
        n_iterations (int, optional): Number of iterations to run for
            each combination of simulation and estimation model.
            Defaults to 5.
        trained_model_dir (str, optional): Directory path where trained
            models are saved. Defaults to None.
        n_samples (int, optional): Number of samples for parameter
            estimation. Defaults to 500.
        n_subjects (int, optional): Number of subjects in simulated data.
            Defaults to 200.
        n_jobs (int, optional): Number of jobs to run in parallel.
            Defaults to -1.

    Returns:
        pd.DataFrame: DataFrame containing results from the model recovery.

    Note:
        Ensure all functions called within have appropriate definitions and
        imports in your script.
    """

    # Calculate how many subjects to simulate based on iterations and subjects
    # per iteration
    n_subs_simulate = n_iterations * n_subjects

    # Generate simulation parameters and unpack the returned values.
    (
        second_stage_states_all,
        rewards_all,
        reward_probs_all,
        available_side_all,
        available_side,
        starting_value_estimate,
        starting_transition_prob_estimate,
        params_dict,
        true_params_array,
    ) = generate_simulation_parameters(n_subs_simulate, task_spec_path)

    # Specify models
    models = ["mf_only", "mb_only", "weighting", "weighting_fixed"]

    # Dictionary to store simulated data
    simulated_data_dict = {}

    # Parameters that are common across all model types
    common_params = [
        params_dict["temperature"],
        starting_value_estimate,
        starting_transition_prob_estimate,
        second_stage_states_all,
        reward_probs_all,
        rewards_all,
        available_side_all,
    ]

    # Loop through models and simulate data
    for model in models:
        simulated_data_dict[model] = simulate_model_with_params(
            model, params_dict, available_side, common_params, random_seed=100
        )

    # Load estimation models from trained_model_dir
    estimation_models = find_trained_models(models, trained_model_dir)

    # Prepare combinations for parallel processing
    combinations = [
        (
            simulated_model,
            simulated_data,
            estimation_model,
            estimation_trained_model,
            iteration,
        )
        for simulated_model, simulated_data in simulated_data_dict.items()
        for estimation_model, estimation_trained_model in estimation_models.items()
        for iteration in range(n_iterations)
    ]

    # Prepare a template array for sampled parameters.
    # This template is used across all parallel jobs and passed to the
    # `run_combination` function.
    sampled_param_template = np.zeros((n_samples, n_subjects, 6))

    # Check if parallel processing should be used based on `n_jobs`
    if n_jobs != 1:
        # Run parallel computation over all combinations using joblib
        with tqdm_joblib(tqdm(desc="Model recovery", total=len(combinations))):
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(run_combination)(
                    simulated_model,
                    simulated_data,
                    estimation_model,
                    estimation_trained_model,
                    iteration,
                    n_subjects,
                    n_samples,
                    available_side,
                    second_stage_states_all,
                    rewards_all,
                    reward_probs_all,
                    sampled_param_template,
                )
                for (
                    simulated_model,
                    simulated_data,
                    estimation_model,
                    estimation_trained_model,
                    iteration,
                ) in combinations
            )
    else:
        # Run the computation serially without joblib
        results_list = [
            run_combination(
                simulated_model,
                simulated_data,
                estimation_model,
                estimation_trained_model,
                iteration,
                n_subjects,
                n_samples,
                available_side,
                second_stage_states_all,
                rewards_all,
                reward_probs_all,
                sampled_param_template,
            )
            for (
                simulated_model,
                simulated_data,
                estimation_model,
                estimation_trained_model,
                iteration,
            ) in combinations
        ]
    # Concatenate all results into a single DataFrame
    results = pd.concat(results_list, axis=0).reset_index(drop=True)

    return results
