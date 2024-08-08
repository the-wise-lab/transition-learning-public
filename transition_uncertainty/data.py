import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from .analysis_utils import calculate_performance


def load_and_process_data(
    sample: str,
    scale: bool = True,
    transform: bool = True,
    include_confidence: bool = True,
    include_questionnaire: bool = True,
    include_factor: bool = True,
    include_param: bool = True,
    include_performance: bool = True,
    retain_unscaled: List[str] = [],
) -> pd.DataFrame:
    """
    Load and prepare datasets for analysis based on specified parameters.

    Args:
        sample (str): The type of sample to load ('discovery', 'replication',
            'follow-up', 'follow-up-1yr', 'retest'). 
        scale (bool): Whether to scale the data. Defaults to True. 
        transform (bool): Whether to transform the variance variable.
            Defaults to True. 
        include_confidence (bool): Whether to include confidence data. 
            Defaults to True. 
        include_questionnaire (bool): Whether to include questionnaire data. 
            Defaults to True. include_factor
        (bool): Whether to include factor data. Defaults to True. 
        include_param (bool): Whether to include param data. Defaults to True.
        include_performance (bool): Whether to include performance data.
            Defaults to True. 
        retain_unscaled (List[str], optional): List of columns to retain 
            in the unscaled DataFrame. For these columns, values will be scaled
            (if scale == True), but an additional column will be created with
            the suffix "_unscaled" which retains the unscaled data. Defaults to
            [].

    Returns:
        pd.DataFrame: The prepared DataFrame after loading, merging, filtering,
        and optionally transforming and scaling.
    """
    # Base file path
    base_path = f"data/{sample}"

    # Load common datasets
    qdata = pd.DataFrame()

    if include_questionnaire:
        qdata = pd.read_csv(
            f"{base_path}/questionnaires/questionnaire_data-reverse_coded-subscales.csv"
        )

    if include_factor:
        factor_df = pd.read_csv(
            f"results/{sample}/transdiagnostic-factors/predicted_factor_scores.csv"
        )
        qdata = (
            factor_df
            if qdata.empty
            else qdata.merge(factor_df, on="subjectID", how="outer")
        )

    if include_param:
        param_df = pd.read_csv(
            f"results/{sample}/transition-task_model-fit/param_df.csv"
        )
        qdata = (
            param_df
            if qdata.empty
            else qdata.merge(param_df, on="subjectID", how="outer")
        )

    if include_confidence:
        confidence_param_df = pd.read_csv(
            f"results/{sample}/confidence_model-fit//confidence_param_df.csv"
        )
        qdata = (
            confidence_param_df
            if qdata.empty
            else qdata.merge(confidence_param_df, on="subjectID", how="outer")
        )

    if include_performance:
        performance = calculate_performance(
            f"{base_path}/transition-task/cannonball_task_data.csv"
        )
        qdata = (
            performance
            if qdata.empty
            else qdata.merge(performance, on="subjectID", how="outer")
        )

    # Load and merge two-step data if sample is 'replication'
    if sample in ["replication", "retest"]:
        two_step = pd.read_csv(
            f"results/{sample}/two-step_model-fit/twostep_subject_effects.csv"
        )
        two_step = two_step.rename(columns={"parameter": "two_step_parameter"})
        qdata = qdata.merge(two_step, on="subjectID")

    initial_subject_count = len(qdata)
    print(f"Initial number of subjects: {initial_subject_count}")

    # Data filtering
    if (
        "gender" in qdata.columns
    ):  # for retest data we don't have questionnaire measures
        # Filter Data: Exclude specific categories and subjects based on
        # conditions. Exclude gender == 2 or 3 (other, no response) - we only
        # have a very small number, which is not enough to properly estimate
        qdata = qdata[qdata["gender"] < 2]
        # Check we only have 0 or 1 gender
        assert qdata["gender"].isin([0, 1]).all()

        # print number of subjects
        print(
            f"Number of included subjects after excluding gender != 0 or 1: {len(qdata)}"
        )

        # Check that ages aren't too extreme
        assert (qdata["age"] >= 18).all()
        assert (qdata["age"] < 100).all()

        # print number of subjects
        print(
            f"Number of included subjects after excluding age < 18 or >= 100: {len(qdata)}"
        )

        # Excluding subjects who fail attention checks.
        qdata = qdata[qdata["attention_check_failed"] == 0]

        # Excluding subjects who fail more than one infrequency item check.
        qdata = qdata[qdata["inattentive_incorrect"] <= 1]

        # print number of subjects
        print(
            f"Number of included subjects after excluding attention check failures: {len(qdata)}"
        )

        # Print demographics
        print(
            "Mean (SD) age = {:.2f} ({:.2f})".format(
                qdata["age"].mean(), qdata["age"].std()
            )
        )
        print(
            "n male = {0}; n female = {1}".format(
                (qdata["gender"] == 0).sum(), (qdata["gender"] == 1).sum()
            )
        )

        # 10 as a cutoff for "moderate" GAD (Spitzer et al., 2006)
        print(
            "Number scoring above cutoff for GAD: {} ({}%)".format(
                (qdata["GAD_7"] >= 10).sum(),
                np.round((qdata["GAD_7"] >= 10).sum() / len(qdata) * 100, 2),
            )
        )

        # 10 as a cutoff for MDD (Kroenke et al., 2009)
        print(
            "Number scoring above cutoff for MDD: {} ({}%)".format(
                (qdata["PHQ_8"] >= 10).sum(),
                np.round((qdata["PHQ_8"] >= 10).sum() / len(qdata) * 100, 2),
            )
        )

    # Data transformation
    if transform:
        qdata["transition_var"] = np.exp(qdata["transition_var"] * 100)

    # Data scaling
    if scale:
        # Duplicate columns to retain unscaled data
        for col in retain_unscaled:
            qdata[f"{col}_unscaled"] = qdata[col]

        # Create a list of columns not to scale
        cols_to_exclude = ["subjectID", "gender"] + [
            i + "_unscaled" for i in retain_unscaled
        ]

        # Scale
        cols_to_scale = [
            col for col in qdata.columns if col not in cols_to_exclude
        ]
        qdata[cols_to_scale] = (
            qdata[cols_to_scale] - qdata[cols_to_scale].mean(numeric_only=True)
        ) / qdata[cols_to_scale].std()

    print(f"Number of subjects after filtering and processing: {len(qdata)}")

    return qdata
