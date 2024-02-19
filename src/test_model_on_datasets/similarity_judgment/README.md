# Batch Similarity Judgment Analysis

This directory contains the necessary files to perform a batch similarity judgment analysis across multiple datasets. It includes various `toml` configuration files, each representing a unique comparison to be performed.

> **Note**: Multiple comparisons can be conducted on a single dataset. For instance, the `CSE_CIE_dots` dataset includes four categories: `linearity`, `orientation`, `proximity`, and `single`. A separate similarity judgment analysis is performed for each category, and the results can be combined to determine the Configural Superiority Effect (CSE). For example, you can compare the "discriminability" between image pairs under the `linearity` condition versus the `single` condition to verify the presence of CSE. This approach is similar to the one used in [Mixed Evidence For Gestalt Grouping in DNN](https://link.springer.com/article/10.1007/s42113-023-00169-2).

For instructions on how to perform the similarity judgment analysis _on individual datasets_, including details about the `toml` configuration file used to run the analysis, please refer to the [documentation here](../../utils/similarity_judgment/README.md).

## Results

The results can be found in [`results/similarity_judgment'](../../../results/similarity_judgments/). They are generated as follows:

- [`compute_similarity_judgments.py`](./compute_similarity_judgments.py) is executed, which feeds each `toml` file to the similarity judgment module, generating a dataframe for each file.
- [`generate_report.py`](../../utils/similarity_judgment/generate_report.py) is then run. By default, it generates a report for any `dataframe.csv` file it finds in `result/similarity_judgments`, creating succinct reports. Each report is located in the same folder as the `dataframe.csv` file used to generate it.
