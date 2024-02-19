import argparse
import pathlib
from src.utils.similarity_judgment.analysis import run_standard_analysis_one_layer
import papermill as pm
import glob
from tqdm import tqdm

dataframe_path = "results/similarity_judgments/contour_completion/dataframe.csv"


def generate_report(result_csv_path=None):
    parameters = {
        "dataframe_path": result_csv_path,
        "idx_layer_used": -1,
    }

    report_folder = pathlib.Path(result_csv_path).parent / "report"
    report_folder.mkdir(exist_ok=True, parents=True)
    try:
        pm.execute_notebook(
            "src/utils/similarity_judgment/report_template.ipynb",
            str(report_folder / "report.ipynb"),
            parameters=parameters,
            progress_bar=False,
        )
    except Exception as e:
        print(e)
        print(
            "Could not generate the report. This might be due to a non-standard format for the analysis. We suggest in this case to run your own custom anaylsis"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="result_csv_path",
        help="The path to the dataframe that we want to generate the report of.",
    )

    args = parser.parse_known_args()[0]
    generate_report(**args.__dict__)
