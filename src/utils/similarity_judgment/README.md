# Similarity Judgment Analysis

## Method Overview

This method allows the analysis of the internal activation of a network when fed with different images. Images are always compared in pairs. Each image is fed individually, the internal activation for the requested layers are extracted, and then the two sets of activations are compared using either the Euclidean Distance or the Cosine Similarity (other comparisons technique might be added in the future).


## Getting Started
The best way to get started is to take a look at the provided example in [`the example folder`](/home/ft20308/mind-set/src/utils/similarity_judgment/examples). Here the Similarity Judgment Method is applied to the `Relational vs Coordinate dataset`. Just running the script [relational_vs_coordinate.py](examples/relational_vs_coordinate.py) as a module (as usual, from the root folder) should be enough: it will use [`this config file`](examples/generate_relational_vs_coordinate_dataset.toml) to compute the Euclidean Distance for many pair of images on the corresponding dataset (using its `annotation.csv` file), generating a report in `examples/results/relational_vs_coordinate`. 
Note that if the dataset doesn't exist, it will create it using [this other config file](examples/generate_relational_vs_coordinate_dataset.toml).

In brief, the way this works is that each level of the indicated `factor_variable` will be compared against each other level. Stimuli will be matched according to the `match_factors` parameters. 

Let's consider the `Relational vs Coordinate` dataset. Once the dataset is created, we can open the corresponding `annotation.csv` file to see the different factors the stimnuli are grouped by. It will look like this (with more columns which we can ignore for this example)

| Path                  | Class           | Id |
|-----------------------|-----------------|-------
| coordinate_change/1.png | coordinate_change|  1 | 
| coordinate_change/2.png | coordinate_change|  2 | 
| coordinate_change/3.png | coordinate_change|  3 |
...
| relation_change/1.png   | relation_change |  1 | 
| relation_change/2.png   | relation_change |  2 | 
| relation_change/3.png   | relation_change |  3 | 
...
| basis/1.png            | basis           |  1 |     
| basis/2.png            | basis           |  2 |     
| basis/3.png            | basis           |  3 |     
...



 The `coordinate`, `relation` and `basis` type of stimuli are grouped under the factor `Class`. 
 Note how each of the `Class` level contains 6 images (specified by the `Id` factor - only 3 showed in the table above). 
The user indicates against which reference level the other levels needs to be compare: so specifying `factor_variable="Class"` and `reference_level = "basis"` will result in images from the `coordinate` level and in the `metric` to be compared against each image in the `basis` level. If we want to make sure that only the images with the same ID are compared (for example, the basis image `Id=3` with the coordinate change image `Id=3`), we need to specify `match_factors = ["Id"]`. Note that you can have multiple matching factors, or None. The [defeault_distance_config](default_distance_config.toml) file contains a complete documentation of each configuration parameter. The generated report works best when there are at least 3 leves for the factor of interest (in which one is the reference level). In this way, you will have at least 2 samples to compare (in the form of boxplots). 

Not all type of similarity analysis might match this structure. For example, the `texturized_blob` (another [example](/home/ft20308/mind-set/src/utils/similarity_judgment/examples)) involves a slightly more complex setup and analysis. 


## Code Overview

### Main Function

The main function is `compute_distance` in `srcutils/similarity_judgment/run.py`. This function requires a set of parameters that can be provided through a `toml` file (e.g. [`this example toml file`](./examples//sim_judgm_relational_vs_coordinate.toml)). The `toml` file will refer to an `annotation.csv` file, which is automatically created when a dataset is generated. For example, the example toml file refers to the [`examples/data/low_mid_level_vision/relational_vs_coordinate/annotation.csv`](../../../examples/data/low_mid_level_vision/relational_vs_coordinate/annotation.csv) file.

### Configuration Parameters

The `toml` file for the similarity judgment method can have many parameters. You can see _all_ parameters in the [default_distance_config.toml](default_distance_config.toml). This file includes all the explanation of what each parameter does. When you run `compute_distance` providing your own toml file, any parameter that is _not_ included in your `toml` file will default to the parameter specified in the [`default_distance_config.toml`](default_distance_config.toml). In this way, your config file can be kept relatively short, and only contain the relevant parameters. 

## Create a Report

To generate a succinct report based on the data collated in the `dataframe.csv` file, execute the command `python -m src.utils.generate_report --results_csv_path path/to/dataframe.csv`. If the `--results_csv_path` parameter is omitted, the script defaults to identifying all `*/dataframe.csv` files in the `results/similarity_judgments` folder, generating individual reports for each. These reports comprise a set of images and a `.ipynb` file, conveniently stored in the same directory as the respective `dataframe.csv` file.
This default report is particularly useful when you run a similarity judgments analysis in which the interested factor has at least 3 levels.