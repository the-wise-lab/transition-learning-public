from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from model_fit_tools.plotting import (
    plot_pp,
    plot_recovery,
    plot_recovery_matrix,
)


def plot_timepoint_correlations(
    data: pd.DataFrame,
    variables: List[str],
    fig_kwargs: Dict = {},
    regplot_kwargs: Dict = {},
    palette: Optional[List[str]] = None,
):
    """
    Plots correlation subplots for given variables.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        variables (List[str]): List of variables (as strings) to plot
            correlations for.
        fig_kwargs (Dict, optional): Keyword arguments for the figure.
            Defaults to `{}`.
        regplot_kwargs (Dict, optional): Keyword arguments for seaborn's
            regplot. Defaults to `{}`.
        palette (Optional[List[str]], optional): List of colours for the
            plots. If `None`, uses the default colour cycle.

    This function initializes a subplot with columns corresponding to the
    number of variables, creates a regression plot for each variable
    showing the correlation between its original and follow-up data, and
    adjusts the layout for clear presentation.
    """
    # Set default values for figure size and dpi if not specified in fig_kwargs
    fig_kwargs.setdefault("figsize", (2.3 * len(variables), 2.5))
    fig_kwargs.setdefault("dpi", 100)

    # Initialize a subplot with columns corresponding to the number of
    # variables
    f, ax = plt.subplots(1, len(variables), **fig_kwargs)

    # Deal with the case where we just have one ax
    if not len(variables) > 1:
        ax = [ax]

    # Use provided color palette or retrieve the default color cycle
    colours = (
        palette
        if palette
        else plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )

    # Define default regplot arguments
    default_regplot_kwargs = {
        "scatter_kws": {"alpha": 0.5},
        "line_kws": {"lw": 2},
    }

    # Update the default regplot arguments with any provided regplot_kwargs
    default_regplot_kwargs.update(regplot_kwargs)

    # Loop through each variable to plot its correlation
    for i, variable in enumerate(variables):
        # Apply regplot_kwargs while ensuring some defaults
        sns.regplot(
            x=variable + "__T1",  # X-axis data column
            y=variable + "__T2",  # Y-axis data column
            data=data,  # Data source
            ax=ax[i],  # Current subplot axis
            color=colours[
                i % len(colours)
            ],  # Colour from the color cycle or provided palette
            **default_regplot_kwargs,
        )

        # Calculate the correlation coefficient
        correlation = (
            data[[variable + "__T1", variable + "__T2"]].corr().iloc[0, 1]
        )

        # Set labels and title for the subplot
        ax[i].set_xlabel("T1")
        ax[i].set_ylabel("T2")
        ax[i].set_title(f"{variable}: r = {correlation:.2f}")

    # Adjust layout
    plt.tight_layout()

    # Return the axes
    return ax


def plot_timepoint_stripplots(
    data: pd.DataFrame,
    variables: List[str],
    fig_kwargs: Dict = {},
    stripplot_kwargs: Dict = {},
    colour_palette: Optional[List[str]] = None,
):
    """
    Plots strip plots for given factors with lines connecting data points
    for each subject.

    Args:
        data (pd.DataFrame): The DataFrame containing the data. Expects
            data in long format, with a column indicating which time point
            the data comes from.
        variables (List[str]): List of variables (as strings) to plot.
        fig_kwargs (Dict, optional): Keyword arguments for the figure.
            Defaults to `{}`.
        stripplot_kwargs (Dict, optional): Keyword arguments for seaborn's
            stripplot. Defaults to `{}`.
        colour_palette (Optional[List[str]], optional): List of colours for
            the plots. If `None`, uses the default colour cycle.

    Each subplot corresponds to a factor, displaying a strip plot with
    lines connecting data points for each subject.
    """
    # Set default values for figure size and dpi if not specified in fig_kwargs
    fig_kwargs.setdefault("figsize", (2.3 * len(variables), 2.3))
    fig_kwargs.setdefault("dpi", 100)

    # Initialise a subplot with columns corresponding to the number of
    # variables
    f, ax = plt.subplots(1, len(variables), **fig_kwargs)

    # Deal with the case where we just have one ax
    if not len(variables) > 1:
        ax = [ax]

    # Use provided colour palette or retrieve the default colour cycle
    colours = (
        colour_palette
        if colour_palette
        else plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )

    # Define default stripplot arguments
    default_stripplot_kwargs = {"s": 3, "jitter": False, "alpha": 0.05}

    # Update the default stripplot arguments with any provided stripplot_kwargs
    default_stripplot_kwargs.update(stripplot_kwargs)

    # Loop through each factor to plot its data
    for i, variable in enumerate(variables):
        # Filter data for the current factor
        variable_data = data[data["variable"] == variable]

        # Apply stripplot_kwargs while ensuring some defaults
        sns.stripplot(
            x="timepoint",
            y="value",
            data=variable_data,
            ax=ax[i],
            color=colours[
                i % len(colours)
            ],  # Colour from the colour cycle or provided palette
            **default_stripplot_kwargs,
        )

        # Collect lines for each subject in a list
        lines = []
        subjects = variable_data["subjectID"].unique()
        for subject in subjects:
            subject_scores = variable_data[
                variable_data["subjectID"] == subject
            ]["value"]
            line = [(0, subject_scores.iloc[0]), (1, subject_scores.iloc[1])]
            lines.append(line)

        # Create a LineCollection and add it to the axis
        lc = LineCollection(lines, color=colours[i % len(colours)], alpha=0.3)
        ax[i].add_collection(lc)

        # Add variable name as title
        ax[i].set_title(variable)

    # Improve plot appearance
    sns.despine()
    plt.tight_layout()

    # Return the axes
    return ax


def plot_summary_data(
    summary: pd.DataFrame,
    var_names: List[str],
    target_var: str,
    datasets: List[str],
    x_labels: List[str],
    ax: plt.Axes,
    alpha: float = 0.05,
    significance_thresholds: Dict[float, str] = {
        0.001 / 3: "***",
        0.01 / 3: "**",
        0.05 / 3: "*",
    },
    star_size: int = 10,
    star_y_offset: float = 0.00,
    spacing: float = 0.2,
    legend: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    ylabel: str = r"$\beta$",
    title: str = "",
    palette: Optional[List[str]] = None,
    scatter_kwargs: Optional[Dict] = None,
    errorbar_kwargs: Optional[Dict] = None,
) -> None:
    """
    Plots parameter estimates with error bars and significance markers
    based on a dataframe including summaries of regression models across
    multiple datasets.

    Args:
        summary (pd.DataFrame): Dataframe containing the summary statistics.
        var_names (List[str]): List of variable names used as predictors.
        target_var (str): Name of the target variable.
        datasets (List[str]): List of datasets to plot (e.g., `['discovery',
            'replication']`).
        x_labels (List[str]): Labels for the x-axis corresponding to each
            variable.
        ax (plt.Axes): Matplotlib Axes object where the plot will be drawn.
        alpha (float, optional): Significance level used for testing,
            defaults to `0.05`.
        significance_thresholds (Dict[float, str], optional): Mapping of
            p-value thresholds to significance symbols.
        star_size (int, optional): Font size for significance stars.
        star_y_offset (float, optional): Vertical offset for significance
            stars.
        spacing (float, optional): Horizontal spacing between groups of
            data points.
        legend (bool, optional): Whether to display a legend.
        ylim (Tuple[float, float], optional): Limits for the y-axis.
        ylabel (str, optional): Label for the y-axis.
        title (str, optional): Title for the plot.
        palette (List[str], optional): List of colours for the plot. If
            `None`, uses the default colour cycle.
        scatter_kwargs (Dict, optional): Additional keyword arguments for
            `ax.scatter`.
        errorbar_kwargs (Dict, optional): Additional keyword arguments for
            `ax.errorbar`.

    """
    # Default keyword arguments for scatter and errorbar functions
    scatter_defaults = {
        "edgecolors": "black",
        "linewidths": 1,
    }
    errorbar_defaults = {
        "capsize": 0,
        "capthick": 2,
        "linewidth": 2,
        "zorder": -1,
        "alpha": 0.8,
    }

    # Update defaults with any user-provided keyword arguments
    if scatter_kwargs:
        scatter_defaults.update(scatter_kwargs)
    if errorbar_kwargs:
        errorbar_defaults.update(errorbar_kwargs)

    # Retrieve the default color palette from matplotlib for consistent styling
    if palette is None:
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Get number of elements for each x axis location
    num_elements = len(datasets)

    # Loop through each variable and dataset to plot
    for i, v in enumerate(var_names):
        for j, dataset in enumerate(datasets):
            # Extract the necessary statistics for each variable and dataset
            beta = summary.loc[
                (summary["dataset"] == dataset)
                & (summary["predictor"] == v)
                & (summary["target"] == target_var),
                "coef",
            ].values[0]
            ci_low = summary.loc[
                (summary["dataset"] == dataset)
                & (summary["predictor"] == v)
                & (summary["target"] == target_var),
                "lower_ci",
            ].values[0]
            ci_high = summary.loc[
                (summary["dataset"] == dataset)
                & (summary["predictor"] == v)
                & (summary["target"] == target_var),
                "upper_ci",
            ].values[0]
            p_value = summary.loc[
                (summary["dataset"] == dataset)
                & (summary["predictor"] == v)
                & (summary["target"] == target_var),
                "p",
            ].values[0]

            # Label datasets only on the first pass to avoid redundancy
            label = dataset.capitalize() if i == 0 else None

            # Calculate x position
            if num_elements == 1:
                x_pos = i
            else:
                x_pos = (
                    i
                    + ((j - (num_elements - 1) / 2) / ((num_elements - 1) / 2))
                    * spacing
                )

            # Plotting the point estimates as scatter points
            ax.scatter(
                x_pos,
                beta,
                color=palette[j],
                label=label,
                **scatter_defaults,
            )

            # Adding error bars for high-density intervals
            ax.errorbar(
                x=[x_pos, x_pos],
                y=[ci_low, ci_high],
                color=palette[j],
                **errorbar_defaults,
            )

            # Annotating the plot with significance stars if applicable
            if p_value < alpha:
                star_string = ""
                for threshold in sorted(
                    significance_thresholds.keys(), reverse=True
                ):
                    if p_value < threshold:
                        star_string = significance_thresholds[threshold]
                ax.text(
                    x_pos,
                    ci_high + star_y_offset,
                    star_string,
                    ha="center",
                    va="bottom",
                    fontsize=star_size,
                    color="#292929",
                )

    # Optionally add a legend
    if legend:
        ax.legend(
            title="Dataset",
            bbox_to_anchor=(1.45, 0.5),
            loc="center",
            frameon=True,
        )

    # Optionally set y-axis limits
    if ylim:
        ax.set_ylim(*ylim)

    # Set x-axis properties
    ax.set_xlim(-0.5, len(var_names) - 0.5)
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Set labels and title
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add a horizontal line at zero for reference
    ax.axhline(0, color="grey", linestyle="--", alpha=0.4)

    # Despine the plot for aesthetics
    sns.despine()


def plot_recovery_results(
    model_name: str,
    true_params_array: np.ndarray,
    sampled_params_dict: Dict[str, np.ndarray],
    plot_recovery_kwargs: Dict = {},
    plot_pp_kwargs: Dict = {},
    plot_recovery_matrix_kwargs: Dict = {},
) -> None:
    """
    Plot results for a specified model.

    The function uses pre-specified parameters for different models,
    extracts the relevant true parameters, and creates a series of plots
    using the extracted true parameters and the provided sampled parameters.

    Args:
        model_name (str): Name of the model to be plotted, must be one of
            `['mf_only', 'mb_only', 'weighting', 'weighting_fixed']`.
        true_params_array (np.ndarray): Array of true parameters.
        sampled_params_dict (Dict[str, np.ndarray]): Dictionary of sampled
            parameters for each model.
        plot_recovery_kwargs (Dict, optional): Additional keyword arguments
            for the plot_recovery function.
        plot_pp_kwargs (Dict, optional): Additional keyword arguments for
            the plot_pp function.
        plot_recovery_matrix_kwargs (Dict, optional): Additional keyword
            arguments for the plot_recovery_matrix function.

    Raises:
        ValueError: If `model_name` is not recognized.
    """

    # Define the parameters to be included in each model.
    model_included_params = {
        "mf_only": ([0, 2, 5], ["tau_value", "decay_value", "temperature"]),
        "mb_only": ([1, 3, 5], ["tau_prob", "decay_prob", "temperature"]),
        "weighting": (
            [0, 1, 2, 3, 4, 5],
            [
                "tau_value",
                "tau_prob",
                "decay_value",
                "decay_prob",
                "W",
                "temperature",
            ],
        ),
        "weighting_fixed": (
            [0, 1, 2, 3, 5],
            [
                "tau_value",
                "tau_prob",
                "decay_value",
                "decay_prob",
                "temperature",
            ],
        ),
    }

    # Validate the model name.
    if model_name not in model_included_params:
        raise ValueError(f"Model name {model_name} not recognized!")

    # Extract relevant indices and parameter names.
    included_param_idx, included_param_names = model_included_params[
        model_name
    ]

    # Extract the corresponding true parameters.
    model_true_params_array = true_params_array.copy()[:, included_param_idx]

    # Plot recovery
    plot_recovery(
        model_true_params_array,
        sampled_params_dict[model_name],
        param_names=included_param_names,
        **plot_recovery_kwargs,
    )

    # Plot PP
    plot_pp(
        model_true_params_array,
        sampled_params_dict[model_name],
        param_names=included_param_names,
        **plot_pp_kwargs,
    )

    # Plot recovery matrix
    plot_recovery_matrix(
        model_true_params_array,
        sampled_params_dict[model_name],
        param_names=included_param_names,
        scale=2,
        xtick_rotation=45,
        **plot_recovery_matrix_kwargs,
    )
