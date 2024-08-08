import argparse
from transition_uncertainty.model_fit_utils import (
    print_timestamped_message,
    train_and_save_model,
    train_model_with_params,
)
from transition_uncertainty.modelling_utils import (
    generate_simulation_parameters,
)

if __name__ == "__main__":
    # Initialize argument parser and add arguments
    parser = argparse.ArgumentParser(
        description="Model Training Argument Parser"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model type to be used. One of 'mf_only', 'mb_only', 'weighting', 'weighting_fixed'",
    )
    parser.add_argument(
        "--n", type=int, help="Number of subjects for simulation"
    )
    args = parser.parse_args()

    # Define the number of subjects based on the parsed argument
    N_SUBS = args.n

    # Path to the task specification data file.
    task_spec_path = "data/task_spec"

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
    ) = generate_simulation_parameters(N_SUBS, task_spec_path)

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

    # Configuration dictionary providing mapping between model types and their
    # specific configurations.
    model_configurations = {
        "mf_only": {
            "true_params_subset": true_params_array[:, [0, 2, 5]],
        },
        "mb_only": {
            "true_params_subset": true_params_array[:, [1, 3, 5]],
        },
        "weighting": {
            "true_params_subset": true_params_array,
        },
        "weighting_fixed": {
            "true_params_subset": true_params_array[:, [0, 1, 2, 3, 5]],
        },
    }

    # Check whether the specified model type (args.model) is in the
    # configurations and initiate the training process if it is.
    if args.model in model_configurations:
        train_model_with_params(
            args.model,
            N_SUBS,
            params_dict,
            model_configurations[args.model]["true_params_subset"],
            available_side,
            common_params,
        )
    else:
        # Notify the user that the specified model type is not valid.
        print_timestamped_message(
            f"Invalid model type '{args.model}' specified. Valid model types are: "
            f"{list(model_configurations.keys())}"
        )
