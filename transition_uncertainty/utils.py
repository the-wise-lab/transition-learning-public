import contextlib
import os
import subprocess
from typing import List

import joblib
import numpy as np
from tqdm import tqdm

os.environ["OUTDATED_IGNORE"] = "1"


def check_directories():
    """
    Checks if the script is being run in the root directory and if the
    required data is present.

    Raises:
        RuntimeError: If the script is not run from the root directory
        or if the 'data' directory is empty.
    """
    # Check if the 'notebooks' directory exists
    if not os.path.isdir("notebooks"):
        # If we're currently in a subdirectory of the "notebooks", move
        # two directories up
        if os.path.isdir("../../notebooks"):
            print("Changing directory to root directory of repository...")
            os.chdir("../..")
        else:
            raise RuntimeError(
                "You must run this notebook from the root directory of the "
                "repository, otherwise paths will break. You are currently "
                "in {}".format(os.getcwd())
            )

    # Check if the 'data' directory exists and is not empty
    if not os.path.isdir("data") or len(os.listdir("data")) == 0:
        raise RuntimeError(
            "You must download the data files from OSF and place them in the "
            "/data directory before running this notebook."
        )

    # Check if the 'figures' directory exists and create it if not
    if not os.path.isdir("figures"):
        os.mkdir("figures")


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
        x (float): The value to transform, in the target
            interval [lower, upper].
        lower (float): The lower bound of the interval.
        upper (float): The upper bound of the interval.

    Returns:
        float: The transformed value in the unit interval [0, 1].
    """
    return (x - lower) / (upper - lower)


def repeat_for_all_subjects(
    array: np.ndarray, n_subs: int, extra_axis: int = None
) -> np.ndarray:
    """
    Repeat the input array for all subjects along a new leading axis.

    Parameters:
    array (np.array): The array to be repeated.
    n_subs (int): The number of subjects for repetition.
    extra_axis (int, optional): Additional axis to repeat
        the array. Default is None.

    Returns:
    np.array: An array repeated for all subjects.
    """
    if extra_axis is not None:
        repeated_array = np.repeat(array[None, ..., None], n_subs, axis=0)
    else:
        repeated_array = np.repeat(array[None, ...], n_subs, axis=0)

    return repeated_array


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into tqdm progress bar given as
    argument

    From
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_last_table_number(filename: str) -> int:
    """
    Reads a markdown file and returns the last table number found.

    Args:
        filename (str): The name of the markdown file.

    Returns:
        int: The last table number or 0 if no tables are found.
    """
    if not os.path.exists(filename):
        return 0

    last_table_number = 0
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("*Table S"):
                try:
                    # Extract table number
                    table_number = int(
                        line.split(".")[0].split("S")[-1].strip()
                    )
                    last_table_number = max(last_table_number, table_number)
                except ValueError:
                    continue
    return last_table_number


def save_markdown_tables(
    tables: List[str],
    captions: List[str],
    filename: str,
    prepend_string: str = "",
    append: bool = False,
) -> None:
    """
    Combines multiple Markdown tables into single Markdown string with
     captions, and exports or appends the result to a Markdown (.md) file.

    Args:
        tables (list[pd.DataFrame]): List of pandas DataFrames to convert to
            Markdown
        captions (list[str]): List of captions for each table.
        filename (str): Name of the output Markdown file (should end
            in .md).
        prepend_string (str): String to prepend to the Markdown string.
        append (bool): If True, append to an existing file. If False,
            overwrite the file.

    Returns:
        None: This function writes to a file and does not return anything.
    """

    # Create an empty string to store the Markdown
    markdown_string = ""

    # Prepend the string if provided
    if prepend_string != "":
        markdown_string += prepend_string + "\n\n"

    # Check if lists are of equal length
    if not len(tables) == len(captions):
        raise ValueError("Captions list must be of the same length as tables")

    # Get the last table number if appending
    start_table_number = get_last_table_number(filename) if append else 0

    # Loop through the dataframes
    for i, table in enumerate(tables):
        table_number = start_table_number + i + 1
        markdown_string += table
        markdown_string += f"\n\n*Table S{table_number}. {captions[i]}*\n\n"

    # Determine the mode for opening the file
    file_mode = "a" if append and os.path.exists(filename) else "w"

    # Inform the user about the file operation
    if file_mode == "a":
        print(f"Appending to existing file: {filename}")
    else:
        print(f"Creating or overwriting file: {filename}")

    # Write or append to the markdown file
    with open(filename, file_mode) as file:
        file.write(markdown_string)

    # Call pandoc to convert to PDF and docx
    # Define the pandoc command for pdf
    pandoc_command_pdf = [
        "pandoc",
        filename,
        "-o",
        filename.replace(".md", ".pdf"),
        "-V",
        "papersize:a4paper",
        "-V",
        "geometry:margin=1in",
    ]

    # Define the pandoc command for docx
    pandoc_command_docx = [
        "pandoc",
        filename,
        "-o",
        filename.replace(".md", ".docx"),
    ]

    # Run the pandoc command
    try:
        subprocess.run(pandoc_command_pdf, check=True)
        print("Conversion to PDF completed successfully.")
        subprocess.run(pandoc_command_docx, check=True)
        print("Conversion to docx completed successfully.")
    except FileNotFoundError:
        print(
            "Warning: Pandoc is not installed or not found in the system's "
            "PATH. Cannot convert to PDF/docx"
        )
    except subprocess.CalledProcessError:
        print(
            "Warning: Pandoc is installed but there was an issue running it."
        )
        raise subprocess.CalledProcessError


def generate_regression_captions(tables_dict: dict, sample: str) -> List[str]:
    """
    Generate and print captions for regression and mediation models.

    Args:
        tables_dict (dict): Dictionary containing all regression tables.
        sample (str): The sample name to be included in the captions.

    Returns:
        List[str]: List of captions for each regression table.
    """
    # Define the legend templates
    mediation_legend = (
        "$\\beta$ = regression coefficient, $\\beta_{SE}$ = standard error "
        "of regression coefficient, $p$ = uncorrected p-value, $CI_{2.5}$ = "
        "95% confidence interval lower bound, $CI_{97.5}$ = 95% confidence "
        "interval upper bound."
    )

    regression_legend = (
        "$\\beta$ = regression coefficient, $\\beta_{SE}$ = standard error "
        "of regression coefficient, $p$ = uncorrected p-value, $p_{corr}$ = "
        "Bonferroni corrected p-value, $CI_{2.5}$ = 95% confidence interval "
        "lower bound, $CI_{97.5}$ = 95% confidence interval upper bound."
    )

    # Create a list to store the captions
    captions = []

    for symptoms, tables in tables_dict.items():
        for dependent_var, table in tables.items():

            # Check if the current table is for mediation
            if "mediation" in symptoms:
                independent_var, mediator, dependent_var = dependent_var.split(
                    ";"
                )
                caption = (
                    f"Coefficients for mediation model testing "
                    f"mediation of the association between "
                    f"{independent_var} and {dependent_var} by "
                    f"{mediator} in the **{sample}** sample. "
                    f"Confidence intervals and significance are "
                    f"determined using bootstrapping. "
                    f"{mediation_legend}"
                )
            else:
                # Adjust the legend if 'p_{corr}' is not in the table
                legend = regression_legend
                if "p_{corr}" not in table:
                    legend = legend.replace(
                        ", $p_{corr}$ = Bonferroni corrected p-value", ""
                    )

                # Determine the independent variables description
                independent_var = (
                    "transdiagnostic symptom dimensions"
                    if "transdiagnostic" in symptoms
                    else "state anxiety and depression"
                )

                # Longitudinal case
                if "longitudinal" in symptoms:
                    independent_var = "change in " + independent_var

                # Edit the dependent variable as needed
                dependent_var = dependent_var.replace("\n", " ").lower()
                dependent_var = dependent_var.replace(
                    "model-free - model-based",
                    "model-free fit - model-based fit",
                )

                # Replace delta with change
                dependent_var = dependent_var.replace(r"$\Delta$", "change")

                caption = (
                    f"Coefficients for regression model predicting "
                    f"{dependent_var} from {independent_var} "
                    f"and covariates in the **{sample}** sample. "
                    f"Confidence intervals and significance are "
                    f"determined using bootstrapping. {legend}"
                )

            captions.append(caption)

    return captions


def caption_and_save_markdown_tables(
    tables_dict: dict, sample: str, filename: str, **kwargs
) -> None:
    """
    Generate captions for regression and mediation models, and save the tables

    Args:
        tables_dict (dict): Dictionary containing all regression tables.
        sample (str): The sample name to be included in the captions.
        **kwargs: Additional keyword arguments to pass to save_markdown_tables.
    """

    # Create the directory of the filename if it doesn't exist
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate the captions
    captions = generate_regression_captions(tables_dict, sample)

    # Extract tables from the dictionary into a list
    tables_list = []

    for symptoms, tables in tables_dict.items():
        for dependent_var, table in tables.items():
            tables_list.append(table)

    # Save the markdown tables
    save_markdown_tables(
        tables_list, captions=captions, filename=filename, **kwargs
    )
