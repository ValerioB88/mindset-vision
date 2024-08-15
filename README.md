# MindSet: Vision
![](https://i.ibb.co/pvTVHKw/0-05254-67.png)     ![](https://i.ibb.co/4SvMvCt/28.png)![](https://i.ibb.co/9N4YVxF/c-0.png)


### TL;DR: Just gimme the datasets!
**[MindSet Large on Kaggle](https://www.kaggle.com/datasets/mindsetvision/mindset)** (~0.5 GB)


**[MindSet Lite on Kaggle](https://www.kaggle.com/datasets/mindsetvision/mindset-lite)**  (~150 MB)


## Overview
The `MindSet: Vision` datasets are designed to facilitate the testing of DNNs against controlled experiments in psychology. `MindSet: Vision` datasets focus on a range of low-, middle-, and high-level visual findings that provide important constraints for computational theories. It also provides materials for DNN testing and demonstrates how to evaluate a DNN for each experiment using DNNs pretrained on ImageNet.



## Datasets

`MindSet: Vision` datasets are divided into three categories: `low_mid_level_vision`, `visual_illusions`, and `shape_and_object_recognition`. Each of this category contains many datasets. You can explore and download the datasets in Kaggle.

**A detailed description of each dataset can be found in the related paper [here](https://openreview.net/forum?id=bAaM8cKoMl#discussion)**: refer to Section 2 for an overview, or to Appendix C for more detailed information, including the psychological significance of each dataset, references to relevant papers, and details on the structure of each dataset.

The [TOML file](generate_all_datasets.toml) contains the summary of each dataset together with all the configurable parameters. If you want to take a look at sample images from every dataset, we recommend you check [this HTML page](https://htmlpreview.github.io/?https://github.com/MindSetVision/mindset-vision/blob/master/tests/small_black_bg/summary.html).

The datasets are structured into subfolders (conditions), which are organized based on the dataset's specific characteristics. At the root of each dataset, there's an `annotation.csv` file. This file lists the paths to individual images (starting from the dataset folder) along with their associated parameters. Such organization enables users to use the datasets either exploting their folder structure (e.g. through PyTorch's  ImageFolder) or by directly referencing the annotation file.

 In our provided Decoder, Classification and Similarity Judgment methods we always use the `annotation.csv` approach.

 

### Ready-To-Download Version

`MindSet: Vision` is model-agnostic and offers flexibility in the way each dataset is employed. Depending on the testing method, you may need a few samples or several thousand images. To cater to these needs, we provide two variants of the dataset on Kaggle:

- [Large Version](https://www.kaggle.com/datasets/valerio1988/mindset) with ~5000 samples for each condition.
- [Lite Version](https://www.kaggle.com/datasets/valerio1988/mindset-lite) with ~100 samples for each condition.


Both versions of the `MindSet: Vision` dataset are structured into folders, each containing a specific dataset. Due to Kaggle's current limitations, it's not possible to download these folders individually. Hence, if you need access to a specific dataset, you'll have to download the entire collection of datasets. Alternatively, you can generate the desired dataset on your own following the provided guidelines in the next section.

Similarly, if your research or project requires datasets with more than the provided samples, you can regenerate the datasets with a specific sample size. 

# Generate datasets from scratch
We provide an intuitive interface to generate each dataset from scratch, allowing users to modify various parameters. To ensure compatibility, we recommend using Python 3.10, as this is the version we have tested our project with.

Before proceeding, we suggest creating a new conda environment and installing all dependencies as follows:

```bash
# Create a new conda environment named 'myenv' with Python 3.10
conda create --name myenv python=3.10

# Activate the newly created environment
conda activate myenv

# Install all required dependencies
pip install -r requirements.txt
```
Replace 'myenv' with your preferred environment name. This setup ensures that you have a clean environment specifically configured for this project. Given that MindSet: Vision has been rigorously tested with Python 3.10, we strongly advise using this Python version.


There are two ways to generate the datasets: through a TOML file (which allows for batch generation of many datasets in one go), or by running the script for each dataset (as a a module) passing all arguments in the command line. The TOML approach is the reccomended one. 


<details><summary><b>TOML generation</b></summary>
<br>

You can either generate one single dataset with this approach, or several (all) of them. 
This is done through a TOML configuration file, a simple text file that specifies what datasets should be generated, and what parameters to use for each one of them. The TOML file used to generate the lite and the full version uploaded to Kaggle are provided in the root folder: [`generate_all_datasets.toml`](generate_all_datasets.toml) and [`generate_all_datasets_lite.toml`](generate_all_datasets_lite.toml).
The file contains a series of config options for each dataset. For example, the dataset `un_crowding` in the category `low_mid_level_vision` is specified as:

```toml
["low_mid_level_vision/un_crowding"]
# The size of the canvas. If called through command line, a string in the format NxM eg `224x224`.
canvas_size = [ 224, 224,]
# Specify the background color. Could be a list of RGB values, or `rnd-uniform` for a random (but uniform) color. If called from command line, the RGB value must be a string in the form R_G_B
background_color = [ 0, 0, 0,]
# Specify whether we want to enable antialiasing
antialiasing = false
# What to do if the dataset folder is already present? Choose between [overwrite], [skip]
behaviour_if_present = "overwrite"
# The number of samples for each vernier type (left/right orientation) and condition. The vernier is places inside a flanker.
num_samples_vernier_inside = 5000
# The number of samples for each vernier type (left/right orientation) and condition. The vernier is placed outside of the flankers
num_samples_vernier_outside = 5000
# Specify whether the size of the shapes will vary across samples
random_size = true
# The folder containing the data. It will be created if doesn't exist. The default will match the folder structure of the generation script 
output_folder = "data/low_mid_level_vision/un_crowding"
```

To regenerate datasets:

1. We suggest to not modify the original TOML files but duplicate them: create a file  `my_datasets.toml`.
2. Copy over from `generate_all_datasets.toml` only the config options for the datasets you want to generate (for example, the code block above to generate just the Crowding/Uncrowding dataset) .
3. Adjust parameters in the config as needed. For example set `num_samples_vernier_inside = 50` and `num_samples_vernier_outside=50` for a quick test.  
4. From the root directory, execute `python -m src.generate_datasets_from_toml my_datasets.toml`.

The generated dataset will be saved in the `output_folder` specified in the TOML file.
Of course, you can simply run `python -m src.generate_datasets_from_toml generate_all_datasets.toml` to regenerate all dataset with default parameters. Note that this will take a while, as there are 30+ datasets! 

</details>
<br>
<details>
<summary> <b>Command Line Generation </b></summary>
<br>
From the root folder, call the desired script. For example:

```python
python -m src.generate_datasets.visual_illusions.ebbinghaus_illusion.generate_dataset
```
to generate the Ebbinghaus Illusion with default parameters in the default folder (`data/visual_illusions/ebbinghaus_illusion`). Add `-h` at the end of the line above to see all possible command line arguments:

```
--output_folder OUTPUT_FOLDER, -o OUTPUT_FOLDER
                    The folder containing the data. It will be created if doesn't exist. The default will match the folder structure of the generation script
--canvas_size CANVAS_SIZE, -csize CANVAS_SIZE
                    The size of the canvas. If called through command line, a string in the format NxM eg `224x224`.
--background_color BACKGROUND_COLOR, -bg BACKGROUND_COLOR
                    Specify the background color. Could be a list of RGB values, or `rnd-uniform` for a random (but uniform) color. If called from command line, the RGB value must be a
                    string in the form R_G_B
--antialiasing, -antial
                    Specify whether we want to enable antialiasing
--behaviour_if_present BEHAVIOUR_IF_PRESENT, -if_pres BEHAVIOUR_IF_PRESENT
                    What to do if the dataset folder is already present? Choose between [overwrite], [skip]
--num_samples_scrambled NUM_SAMPLES_SCRAMBLED, -nss NUM_SAMPLES_SCRAMBLED
                    How many samples to generated for the scrambled up conditions
--num_samples_illusory NUM_SAMPLES_ILLUSORY, -nsi NUM_SAMPLES_ILLUSORY
                    How many samples to generated for the illusory conditions (small and big flankers)
```

Specify the ones you wish to modify:

```python
python -m src.generate_datasets.visual_illusions.ebbinghaus_illusion.generate_dataset --num_samples_scrambled 500000
```

Note: Due to the way the datasets are organized, `-h` won't report the default values for the arguments. However all defaults are showed in the [`TOML file`](generate_all_datasets.toml).
</details>
<br>
You might want to generate the same dataset multiple times using the same TOML file. To do that, follow these instructions:
<br>

<p>
<details> <summary> <b>Generate the same dataset multiple times using TOML </b></summary> 

## 
If you need to generate the same dataset multiple times, each with different configurations or parameters, you can include multiple configurations for the same dataset within a single TOML file. However, TOML requires each table (denoted by names within [square brackets]) to have a unique name. To accomplish this, you can use different suffixes or identifiers for each configuration, like so:
```TOML
["low_mid_level_vision/un_crowding.TRAIN"]
...
output_folder = 'blabla/train'

["low_mid_level_vision/un_crowding.EVAL"]
...
output_folder = 'blabla/eval'
```
In this example, `TRAIN` and `EVAL` are distinct identifiers that allow you to define different settings for the same dataset under the `low_mid_level_vision/un_crowding` category. Ensure that the main name remains consistent, as it is used to locate the corresponding dataset generation file in the `src/generate_datasets` folder.
</details>
</p>

# Testing Methods
Although we encourage researchers to use MindSet datasets in a variety of different experimental setups to compare DNNs to humans, we provide the resources to perform a set of basic comparisons between humans and DNNs outputs.
The three methods employed are: 


- **[`Similarity Judgment`](src/utils/similarity_judgment/README.md)**: Compute a distance metric (e.g. `euclidean distance`) between the internal activation of a DNN across different stimulus set. Compare the results with human similarity judgments. In `utils/similarity_judgment`. Works with a wide variety of DNNs. 
- **[`Decoder Method`](src/utils/decoder/README.md)**: Attach simple linear decoders at several processing stages of a ResNet152 network pretrained on ImageNet. The idea is that the researcher trains the decoders on a task (either regression or classification), and then tested on some target condition such an illusory configuration. In `utils/decoder`. 
- **[`ImageNet Classification`](src/utils/imagenet_classification/README.md)**: Test DNNs on unfamiliar data, such as texturized images. in `utils/imagenet_classification`.

Each dataset has a suggested method it could be evaluated on. We provide examples for each method in the corresponding folder.

**Note:** Always set the working directory to the project root (`MindSet`). To manage module dependencies, run the script as a module, e.g., `python -m src.generate_datasets_from_toml generate_all_datasets_lite.toml`.



## Supported Operating Systems
The scripts and functionalities have been tested and are confirmed to work on *macOS 13 Ventura*, *Windows 11*, and *Ubuntu 20.04*.

<!-- ## Similarity Judgement
[![Demo for Similarity Judgement](assets/similarity_judgement.png)](https://youtu.be/a7k5viGmxnk)
 -->


