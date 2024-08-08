from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from behavioural_modelling.learning.beta_models import (
    average_betas,
    leaky_beta_update,
    multiply_beta_by_scalar,
    sum_betas,
)
from behavioural_modelling.utils import choice_from_action_p
from numpy.typing import ArrayLike


@jax.jit
def softmax_difference(
    value1: ArrayLike, value2: ArrayLike, temperature: float
) -> ArrayLike:
    """
    Softmax function that acts upon the difference between two values.

    Args:
        value1 (ArrayLike): Array of values to apply softmax to, of shape
        (n_trials, n_bandits)
        value2 (ArrayLike): Array of values to subtract from value1,
        of shape (n_trials, n_bandits)
        temperature (float): Softmax temperature, in range 0 > inf.

    Returns:
        ArrayLike: Choice probabilities, of shape (2,)
    """
    # Calculate difference between values
    diff = value1 - value2

    # Calculate softmax probabilities
    z = jnp.array([0, -diff]) / temperature
    softmax_output = jnp.exp(z) / jnp.sum(jnp.exp(z))

    return softmax_output


def MB_MF_beta_update_choice_wrapper(
    value_estimate_transition_probs: Tuple[np.ndarray, np.ndarray],
    choices_states_expected_observed_confidence_key: Tuple[
        int, np.ndarray, jax.random.PRNGKey
    ],
    tau_p_value: float,
    tau_n_value: float,
    tau_prob: float,
    decay_value: float,
    decay_prob: float,
    W: float,
    temperature: float,
    n_actions: int,  # STATIC
    simulate: bool,  # STATIC
) -> np.ndarray:
    """
    Wrapper function for model-based/model-free decision-making model with
    leaky beta transition learning to use in jax.lax.scan.

    Assumes a task structure with n starting states and 2 second stage states,
    whereby each starting state transitions to either of the two second stage
    states with probabilites specified by the transition_probs argument.

    Rewards are learnt using a beta learning model, and so it is assumed that
    the rewards are either 0 or 1.

    Args:
        value_estimate_transition_probs (Tuple[np.ndarray, np.ndarray]): Tuple
            of estimated first stage state values and transition probabilities
            (these need to be combined into one argument as scan expects a
            single argument of variables to iterate over). Current model-free
            value estimate are represented by the alpha and beta parameters of
            a beta distribution. An array of shape (2, 2), where the first
            dimension represents the second stage state and the second
            dimension represents the beta distribution parameters. Current
            estimate of transition probabilities (i.e., the probability of
            transitioning to each second stage state from each first stage
            state), represented by the alpha and beta parameters of a beta
            distribution. An array of shape (4, 2), where the first dimension
            represents the first stage state and the second dimension
            represents the beta distribution parameters.
        choices_states_expected_observed_confidence_key (Tuple[float,
            np.ndarray, int, jax.random.PRNGKey]): Tuple of observed choices,
            second stage states, rewards and Jax RNG keys (these need to be
            combined into one argument as scan expects a single argument of
            variables to iterate over). Choices should be provided as an
            integer representing the index of the first-stage option chosen;
            second stage states should be provided as an array indicating which
            second state each first stage option led to on each trial (with
            values of 0 or 1 depending on the second stage state observed).
            Expected and observed rewards should be provided as an array with
            values for each second stage state. Confidence trial indicators are
            provided as an array with a single value for each trial indicating
            which option was available on the confidence trial.
        tau_p_value (float): Positive update rate for value.
        tau_n_value (float): Negative update rate for value.
        tau_prob (float): Update rate for transition probabilities. 
        decay_value (float):Decay rate for value. 
        decay_prob (float): Decay rate for transition probabilities.
        W (float): Weighting of model-based learning contribution 
            relative to model-free learning. 
        temperature (float):Softmax temperature 
        n_actions (int): Number of actions 
        simulate (bool): Whether to simulate or use observed behaviour. 
            If simulating, outputs are based on simulated choices. If not
            simulating, outputs are based on observed choices.
    Returns:
        np.ndarray: Updated value estimate
    """

    # Unpack estimates
    value_estimate, transition_probs = value_estimate_transition_probs

    # Unpack trial outcomes and RNG key
    (
        observed_choice,
        second_stage_states,
        expected_rewards,
        observed_rewards,
        confidence_option,
        key,
    ) = choices_states_expected_observed_confidence_key

    # Calculate intermediate values if simulating
    if simulate:
        # Get mean reward estimate from beta distributions
        mean_value_estimate = value_estimate[:, 0] / (
            value_estimate[:, 0] + value_estimate[:, 1]
        )

        # Get mean transition probability from beta distributions
        mean_transition_probs = transition_probs[:, 0] / (
            transition_probs[:, 0] + transition_probs[:, 1]
        )

        # stack horizontally
        mean_transition_probs = jnp.hstack(
            [mean_transition_probs, 1 - mean_transition_probs]
        )

        # Convert mean_transition_probs to a 2D array so that we have prob and
        # 1-prob
        mean_transition_probs = jnp.vstack(
            [mean_transition_probs, 1 - mean_transition_probs]
        ).T

    else:
        mean_value_estimate = value_estimate
        mean_transition_probs = transition_probs

    # set W on some trials to 0 (i.e., only use MF, as if MB is not available)
    W_on = jnp.array([expected_rewards[0] < 2]).astype(int)

    # Set expected rewards > 1 to 0
    expected_rewards = jnp.where(expected_rewards > 1, 0.5, expected_rewards)

    # Compute MB value estimate for each option Here we compute the MB value
    # estimate for each option by multiplying the transition probabilities by
    # the expected rewards This gives us P(R1) and P(R2) for each option, i.e.
    # the probability of us getting a pink (1) or purple ball (2) AND it not
    # exploding NOTE - here we need to multiply a beta distribution (the
    # probability of a transition) by a scalar probability (the given reward
    # probability). The correct approach here would result in a truncated beta
    # distribution, but this is not helpful for the next step where we need to
    # sum the resulting distributions to get the combined reward probabulity
    # given the probabilties of purple and pink balls. Instead, we multiply the
    # beta distribution by the scalar probability using an approximation that
    # does not accurately represent the shape of the resulting distribution
    # (i.e., the skew is incorrect) but which correctly represents the mean and
    # variance of the distribution. This is enough for our purposes, as we are
    # only interested in the mean and variance of the distribution.
    MB_dist_1_1 = multiply_beta_by_scalar(
        transition_probs, expected_rewards[0]
    )  # P(R1) | action 1
    MB_dist_1_2 = multiply_beta_by_scalar(
        transition_probs[..., ::-1], expected_rewards[1]
    )  # P(R2) | action 1
    MB_dist_2_1 = multiply_beta_by_scalar(
        transition_probs, expected_rewards[1]
    )  # P(R1) | action 2
    MB_dist_2_2 = multiply_beta_by_scalar(
        transition_probs[..., ::-1], expected_rewards[0]
    )  # P(R2) | action 2

    # We then sum these to get total reward probability for each option I.e.,
    # given I choose the left option, what's the combined probability of
    # getting ANY reward (pink or purple)? This works because the transition
    # probabilities are complementary (i.e., P(pink) + P(purple) = 1), so
    # summing these estimates will never result in a mean P > 1
    summed_dist_1 = sum_betas(MB_dist_1_1, MB_dist_1_2)  # P(R) | action 1
    summed_dist_2 = sum_betas(MB_dist_2_1, MB_dist_2_2)  # P(R) | action 2
    # Stack
    summed_dist = jnp.vstack([summed_dist_1, summed_dist_2])

    # Get mean
    if simulate:
        MB_value = summed_dist[:, 0] / (summed_dist[:, 0] + summed_dist[:, 1])
    else:
        MB_value = 0

    # Turn off MB on some trials
    W = W * W_on

    # Need to return something for W_var so set it to 0
    W_var = 0

    # Get weighted MF/MB value estimate
    combined_dists = jnp.vstack(
        [
            average_betas(summed_dist[0, :], value_estimate[0, :], W, 1 - W),
            average_betas(summed_dist[1, :], value_estimate[1, :], W, 1 - W),
        ]
    )

    # Get means
    combined_value_1_mean = combined_dists[0, 0] / (
        combined_dists[0, 0] + combined_dists[0, 1]
    )
    combined_value_2_mean = combined_dists[1, 0] / (
        combined_dists[1, 0] + combined_dists[1, 1]
    )

    # Concatenate
    combined_value = jnp.vstack(
        [combined_value_1_mean, combined_value_2_mean]
    ).T

    # Convert from 0-1 to -1 to 1 - seems to improve recoverability
    combined_value = (combined_value * 1) - 1

    # Get choice probability - using difference seems to work better than
    # softmax on raw values
    choice_p = softmax_difference(
        combined_value_1_mean, combined_value_2_mean, temperature
    )

    # Make a choice
    if simulate:
        choice = choice_from_action_p(key, choice_p, 0)
    else:
        choice = observed_choice

    # Determine whether this a confidence trial (1 = confidence, 0 = normal)
    confidence = jnp.array(confidence_option > -1)
    # If this is a confidence trial, use the option provided
    choice = jnp.array(
        ((1 - confidence) * choice) + (confidence * confidence_option), int
    )[0]

    # Convert to one-hot format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get outcome
    observed_second_stage_state = second_stage_states[choice]
    outcome = observed_rewards[observed_second_stage_state]

    # Get the outcome and update the model-free value estimate
    updated_value = leaky_beta_update(
        value_estimate,
        choice_array,
        outcome,
        tau_p_value,
        tau_n_value,
        decay_value,
        update=jnp.array(
            confidence != 1, int
        ),  # don't update value on confidence trials
    )

    # Get the second stage state and transition probability estimates
    updated_transition_probs = leaky_beta_update(
        transition_probs,
        1,
        second_stage_states[0],
        tau_prob,
        tau_prob,
        decay_prob,
        update=jnp.array(
            second_stage_states[1] != 2, int
        ),  # allows for updating to be turned off
        increment=W_on,
    )

    return (updated_value, updated_transition_probs), (
        value_estimate,
        mean_value_estimate,
        MB_value,
        combined_dists,
        mean_transition_probs,
        transition_probs,
        choice_p,
        choice,
        choice_array,
        W_var,
    )


MB_MF_beta_update_choice_wrapper_jit = jax.jit(
    MB_MF_beta_update_choice_wrapper,
    static_argnums=(9, 10),
)


def MB_MF_beta_trial_choice_iterator(
    key: jax.random.PRNGKey,
    observed_choices: np.ndarray,
    second_stage_states: np.ndarray,
    expected_reward_probs: np.ndarray,
    observed_rewards: np.ndarray,
    confidence_options: np.ndarray,
    n_actions: int,  # STATIC
    n_trials: int,  # STATIC
    starting_value_estimate: float = 1.0,
    starting_transition_prob_estimate: float = 1.0,
    tau_p_value: float = 0.5,
    tau_n_value: float = 0.5,
    tau_prob: float = 0.5,
    decay_value: float = 0.1,
    decay_prob: float = 0.5,
    W: float = 0.5,
    temperature: float = 0.5,
    simulate: bool = False,  # STATIC
) -> np.ndarray:
    """
    Iterate over trials and update value estimates, generating choices for each
    trial. Used for model-fitting.

    Args:
        key (jax.random.PRNGKey): Jax random number generator key
        outcomes (np.ndarray): Trial outcomes for each bandit of shape 
            (n_trials, n_bandits). 
        choices (np.ndarray)
        second_stage_states (np.ndarray)
        expected_reward_probs (np.ndarray): Expected reward probabilities, 
            as a 3D array of shape (n_observations, n_trials, n_bandits).
        rewards (np.ndarray): Rewards associated with each second stage state.
            confidence_options (np.ndarray): The option that was available on
            each confidence trials. Non-confidence trials should be set to -1.
        n_actions (int, optional): Number of actions.
        n_trials (int): Number of trials
        starting_value_estimate (float, optional): Starting value estimate 
            (i.e., the starting value of the A and B parameters
            of a beta distribution). Defaults to 1.
        starting_transition_prob_estimate (float, optional): Starting estimate 
            for state transition probabilities (i.e., the starting value of the
            A and B parameters of a beta distribution). Defaults to 1.
        tau_p_value (float): Positive update rate for value.
        tau_n_value (float): Negative update rate for value.
        tau_prob (float): Update rate for transition probabilities.
        decay_value (float): Decay rate for value.
        decay_prob (float): Decay rate for transition probabilities.
        W (float): Weighting of model-based contribution relative to 
            model-free contribution.
        temperature (float): Softmax temperature
        simulate (bool): Whether to simulate or use observed behaviour. 
            If simulating, outputs are based on simulated choices.
            If not simulating, outputs are based on observed choices.
    Returns:
        np.ndarray: Value estimates for each trial and each bandit
    """

    # Use functools.partial to create a function that uses the same parameter
    # values for all trials
    MB_MF_beta_update_partial = partial(
        MB_MF_beta_update_choice_wrapper_jit,
        tau_p_value=tau_p_value,
        tau_n_value=tau_n_value,
        tau_prob=tau_prob,
        decay_value=decay_value,
        decay_prob=decay_prob,
        W=W,
        temperature=temperature,
        n_actions=n_actions,
        simulate=simulate,
    )

    # Initial values for beta dist parameters
    v_start = jnp.ones((n_actions, 2)) * starting_value_estimate
    t_start = (
        jnp.ones((1, 2)) * starting_transition_prob_estimate
    )  # Only one estimate as probabilities are complementary

    # Jax random keys for choices
    keys = jax.random.split(key, n_trials)

    # use jax.lax.scan to iterate over trials
    _, (
        v,
        mean_value_estimate,
        MB_value,
        combined_dists,
        mean_transition_probs,
        t,
        choice_p,
        choices,
        choices_one_hot,
        W_var,
    ) = jax.lax.scan(
        MB_MF_beta_update_partial,
        (v_start, t_start),
        (
            observed_choices,
            second_stage_states,
            expected_reward_probs,
            observed_rewards,
            confidence_options,
            keys,
        ),
    )

    return (
        v,
        t,
        mean_value_estimate,
        MB_value,
        combined_dists,
        mean_transition_probs,
        choice_p,
        choices,
        choices_one_hot,
        W_var,
    )


# Set up jax JIT and vmaps
MB_MF_beta_trial_choice_iterator_jit = jax.jit(
    MB_MF_beta_trial_choice_iterator, static_argnums=(6, 7, 17)
)

# Vmap to iterate over blocks
MB_MF_beta_simulate_vmap_blocks = jax.vmap(
    MB_MF_beta_trial_choice_iterator_jit,
    in_axes=(
        None,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
)

# Vmap to iterate over observations (subjects)
MB_MF_beta_simulate_vmap_observations = jax.vmap(
    MB_MF_beta_simulate_vmap_blocks,
    in_axes=(0, 0, 0, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
)


def simulate_leaky_beta_transition_learner(
    tau_p_value: np.ndarray,
    tau_n_value: np.ndarray,
    tau_prob: np.ndarray,
    decay_value: np.ndarray,
    decay_prob: np.ndarray,
    W: np.ndarray,
    temperature: np.ndarray,
    starting_value_estimate: float,
    starting_transition_prob_estimate: float,
    second_stage_states: np.ndarray,
    expected_reward_probs: np.ndarray,
    rewards: np.ndarray,
    confidence_options: np.ndarray,
    observed_choices: np.ndarray = None,
    choice_format: str = "index",
    seed: int = 42,
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
    Simulate choices a multi-armed bandit task with n_bandits arms.

    Args:
        tau_p_value (float): Positive update rate for value.
        tau_n_value (float): Negative update rate for value. 
        tau_prob (float): Update rate for transition probabilities. 
        decay_value (float): Decay rate for value. 
        decay_prob (float): Decay rate for transition probabilities. W
        (float): Weighting of model-based and model-free learning. temperature
        (float): Softmax temperature

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]: Choice probabilities, choices, value
        estimates, transition estimates, mean value estimates, model-based
        value estimates, combined value estimates, variance weighting.
    """

    assert (
        tau_p_value.shape
        == tau_n_value.shape
        == tau_prob.shape
        == decay_value.shape
        == decay_prob.shape
        == W.shape
        == temperature.shape
        == starting_value_estimate.shape
        == starting_transition_prob_estimate.shape
    ), "All parameters should have the same shape, but got"
    " {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}".format(
        tau_p_value.shape,
        tau_n_value.shape,
        tau_prob.shape,
        decay_value.shape,
        decay_prob.shape,
        W.shape,
        temperature.shape,
        starting_value_estimate.shape,
        starting_transition_prob_estimate.shape,
    )

    # Extract dimensions
    _, n_blocks, n_trials, n_bandits = second_stage_states.shape
    n_observations = tau_p_value.shape[0]

    # If choices is None, create an array of zeros for choices and set simulate
    # to True
    if observed_choices is None:
        observed_choices = np.zeros(
            (n_observations, n_blocks, n_trials), dtype=int
        )
        simulate = True
    else:
        simulate = False

    # Run simulation
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_observations)
    # subject_keys = jax.random.split(key, alpha_p.shape[0])
    (
        value_estimates,
        transition_estimates,
        mean_value_estimate,
        MB_value,
        combined_dists,
        mean_transition_probs,
        choice_p,
        choices,
        choices_one_hot,
        W_var,
    ) = MB_MF_beta_simulate_vmap_observations(
        keys,
        observed_choices,
        second_stage_states,
        expected_reward_probs,
        rewards,
        confidence_options,
        n_bandits,
        n_trials,
        starting_value_estimate,
        starting_transition_prob_estimate,
        tau_p_value,
        tau_n_value,
        tau_prob,
        decay_value,
        decay_prob,
        W,
        temperature,
        simulate,
    )

    if choice_format == "one_hot":
        return (
            choice_p,
            choices_one_hot,
            value_estimates,
            transition_estimates,
            mean_value_estimate,
            MB_value,
            combined_dists,
            mean_transition_probs,
            W_var,
        )
    elif choice_format == "index":
        return (
            choice_p,
            choices,
            value_estimates,
            transition_estimates,
            mean_value_estimate,
            MB_value,
            combined_dists,
            mean_transition_probs,
            W_var,
        )
