from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from .beta_models import simulate_leaky_beta_transition_learner


def optim_func(
    params: Dict[str, float],
    best_option: np.ndarray,
    simulate_params_beta: List[np.ndarray],
    available_side: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Optimisation function for the model-based beta learning model.

    This takes candidate parameters and calculates the likelihood of choosing
    the best choice on a given trial.

    Args:
        params: A dictionary of parameters to be optimised. These should be:
            tau_prob, decay_prob, temperature.
        best_option: The best option (i.e., the option that leads to a rewarded
            second stage state).
        simulate_params_beta: A list of parameters to be passed to the
            simulation function.
        available_side: The available side for each trial. Used to filter out
            confidence trials.

    Returns:
        Tuple of the negative score, the choice probabilities, and the
        transition estimates (as beta distributions).

    """

    # Unpack the parameters
    tau_prob, decay_prob, temperature = params

    # Update the parameters
    simulate_params_beta[2] = np.ones(1) * tau_prob
    simulate_params_beta[4] = np.ones(1) * decay_prob
    simulate_params_beta[5] = np.ones(1)
    simulate_params_beta[6] = np.ones(1) * temperature

    # Call the simulation function
    choice_p_beta, _, _, transition_estimates_beta, _, _, _, _, _ = (
        simulate_leaky_beta_transition_learner(*simulate_params_beta, seed=42)
    )

    # Remove confidence trials
    choice_p_beta = choice_p_beta[..., available_side[0, :] == -1, :].squeeze()

    # Score
    score = (choice_p_beta * best_option).sum()

    return -score, choice_p_beta, transition_estimates_beta


# Clamping function to enforce bounds
def clamp_params(
    params: jnp.ndarray, bounds: List[Tuple[float, float]]
) -> jnp.ndarray:
    """
    Clamp parameters to the specified bounds.

    Args:
        params: The parameters to be clamped.
        bounds: A list of tuples specifying the lower and upper bounds for each
            parameter.

    Returns:
        The clamped parameters.
    """

    clamped_params = []
    for param, (lower, upper) in zip(params, bounds):
        clamped_params.append(jnp.clip(param, lower, upper))
    return jnp.array(clamped_params)


# Optimization setup
def optimize_with_optax(
    best_option: jnp.ndarray,
    available_side: jnp.ndarray,
    simulate_params_beta: List[jnp.ndarray],
    init_params: jnp.ndarray,
    bounds: List[Tuple[float, float]],
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    tol: float = 1e-5,
    patience: int = 50,
) -> jnp.ndarray:
    """
    Optimize the model using the Optax library.

    Args:
        best_option: The best option (i.e., the option that leads to a rewarded
            second stage state).
        available_side: The available side for each trial. Used to filter out
            confidence trials.
        simulate_params_beta: A list of parameters to be passed to the
            simulation function.
        init_params: The initial parameters for the optimization.
        bounds: A list of tuples specifying the lower and upper bounds for each
            parameter.
        learning_rate: The learning rate for the optimizer.
        num_steps: The maximum number of optimization steps.
        tol: The tolerance for early stopping.
        patience: The number of steps to wait before early stopping.

    Returns:
        The optimized parameters.
    """

    # Define the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)

    # Value and gradient function
    def loss_fn(params):
        return optim_func(
            params, best_option, simulate_params_beta, available_side
        )[0]

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    params = init_params
    best_loss = float("inf")
    no_improvement_steps = 0

    for step in range(num_steps):
        # Compute loss and gradients
        loss, grads = value_and_grad_fn(params)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Clamp parameters to bounds
        params = clamp_params(params, bounds)

        # Check early stopping criteria
        if loss < best_loss - tol:
            best_loss = loss
            no_improvement_steps = 0  # Reset patience counter
        else:
            no_improvement_steps += 1

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.6f}, Best Loss: {best_loss:.6f}")

        # Early stopping condition
        if no_improvement_steps >= patience:
            print(
                f"Early stopping at step {step} with best loss: {best_loss:.6f}"
            )
            break

    return params
