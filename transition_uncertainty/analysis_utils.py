from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pingouin import intraclass_corr


def calculate_intraclass_correlations(
    data: pd.DataFrame, variables: List[str]
) -> pd.DataFrame:
    """
    Calculates intraclass correlations for a list of variables.

    Args:
        data (pd.DataFrame): The DataFrame containing the data variables
        (List[str]): List of variables (as strings) for which to calculate
        intraclass correlations.

    Returns:
        pd.DataFrame: A DataFrame containing intraclass correlation results for
        each variable.

    This function calculates intraclass correlations for each specified
    variable and returns a consolidated DataFrame of the results.
    """
    # Initialize a list to store the intraclass correlation results for each
    # variable
    iccs = []

    # Loop through each variable to compute its intraclass correlation
    for variable in variables:
        # Filter the data for the current variable
        variable_data = data[data["variable"] == variable]

        # Calculate intraclass correlation
        icc_res = intraclass_corr(
            data=variable_data,
            targets="subjectID",
            raters="timepoint",
            ratings="score",
        )

        # Add a column to the result indicating the current variable
        icc_res["variable"] = variable

        # Append the result to the list of intraclass correlations
        iccs.append(icc_res)

    # Concatenate the individual results into a single DataFrame
    iccs = pd.concat(iccs)

    # Filter to only include results of type 'ICC3'
    iccs = iccs[iccs["Type"] == "ICC3"]

    return iccs


def calculate_performance(task_data_path: str) -> pd.DataFrame:
    """
    Calculate performance metrics based on task data.

    This function computes various metrics (mean explosion rate, max score,
    mean absolute confidence, mean bet correctness) per subject based on the
    input task data and then merges them into a single DataFrame.

    Args:
        task_data_path (str): The file path of the task data CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the computed performance metrics
        per subjectID.
    """
    # Read task data from CSV
    task_data = pd.read_csv(task_data_path)

    # Calculate mean explosion rate per subject
    explosions = (
        task_data.groupby("subjectID")
        .mean(numeric_only=True)
        .reset_index()[["subjectID", "exploded"]]
    )

    # Calculate max score per subject
    score = (
        task_data.groupby("subjectID")
        .max()
        .reset_index()[["subjectID", "score"]]
    )

    # Filter data for confidence trials
    confidence_df = task_data[task_data["trialType"] == "confidence"].copy()

    # Calculate mean absolute confidence per subject
    confidence_df["confidence_abs"] = confidence_df["confidence"].abs()
    confidence = (
        confidence_df.groupby("subjectID")
        .mean(numeric_only=True)
        .reset_index()[["subjectID", "confidence_abs"]]
    )

    # Determine the colour bet and whether the bet was correct
    confidence_df["bet_colour"] = np.nan
    confidence_df.loc[confidence_df["pinkBet"] != 0, "bet_colour"] = "pink"
    confidence_df.loc[confidence_df["purpleBet"] != 0, "bet_colour"] = "purple"
    confidence_df["bet_correct"] = (
        confidence_df["bet_colour"] == confidence_df["ballColour"]
    )

    # Calculate mean bet correctness per subject
    confidence_correct_df = (
        confidence_df.groupby("subjectID")
        .mean(numeric_only=True)
        .reset_index()[["subjectID", "bet_correct"]]
    )

    # Merge calculated metrics into a single performance DataFrame
    performance = explosions.merge(score, on="subjectID")
    performance = performance.merge(confidence, on="subjectID")
    performance = performance.merge(confidence_correct_df, on="subjectID")

    return performance


def compile_ols_results(
    model_list: List[sm.regression.linear_model.RegressionResultsWrapper],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compiles results from a list of statsmodels OLS models into a pandas
    DataFrame.

    Args:
        model_list (List[sm.regression.linear_model.RegressionResultsWrapper]):
        A list of statsmodels OLS regression models. 
        alpha (float): Significance level for confidence intervals, 
        defaults to 0.05.

    Returns:
        pd.DataFrame: A DataFrame with columns for the target variable, each
        predictor, coefficients, confidence intervals, and p-values.
    """

    # Initialize an empty list to store the data
    results_data = []

    # Loop through each model in the list
    for model in model_list:
        # Extract the model's target variable name
        target = model.model.endog_names

        # Extract parameter estimates and p-values
        params = model.params
        pvalues = model.pvalues_bootstrap

        # Get upper and lower CI bounds
        conf_intervals = model.conf_int_bootstrap(alpha=alpha)[1:]
        lower_bounds = conf_intervals[0]
        upper_bounds = conf_intervals[1]

        # For each parameter, collect the necessary data
        # Skip the first parameter, which is the intercept
        for predictor, beta in list(params.items())[1:]:
            result = {
                "target": target,
                "predictor": predictor,
                "coef": beta,
                "lower_ci": lower_bounds[predictor],
                "upper_ci": upper_bounds[predictor],
                "p": pvalues[predictor],
            }
            results_data.append(result)

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results_data)
    return results_df
