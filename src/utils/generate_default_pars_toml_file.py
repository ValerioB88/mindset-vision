# %%
"""
This file generates a toml parameters file containing the parameters for each dataset found in the src/generate_datasetss folder (that is, each file matching the path  src/generate_datasets/**/generate_dataset**.py). The toml parameters file is in the format that can be read by the `generate_datasets.py` function. A user is supposed to change the resulting toml file, not the defaults parameters in the individual source files. 
"""

import importlib
import toml
import inspect

import os
from tqdm import tqdm
from pathlib import Path
import sty
import glob

import subprocess
import re
import toml
import inspect

import glob
from pathlib import Path
from tqdm import tqdm

from src.utils.misc import modify_toml


def add_lite(path):
    parts = path.split("/")
    parts[0] += "_lite"
    return "/".join(parts)


def generate_lite(toml_file, output_toml_file):
    # Replace these paths with the actual paths of your files
    with open(toml_file, "r") as file:
        lines = file.readlines()

    new_lines = modify_toml(
        lines,
        modified_key_starts_with="num_samples",
        modify_value_fun=lambda h, x: max(int(x) // 100, 5),
    )
    new_lines = modify_toml(
        new_lines,
        modified_key_starts_with="output_folder",
        modify_value_fun=lambda h, x: add_lite(x),
    )
    with open(output_toml_file, "w") as file:
        file.writelines(new_lines)


def extract_help_text(script_path):
    """Run the script with the -h flag and capture the output."""
    # ".".join(list(Path(script_path).parts)).strip(".py")
    result = subprocess.run(
        ["python", "-m", ".".join(list(Path(script_path).parts)).strip(".py"), "-h"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def add_comments_to_toml(toml_str, section_comments, descriptions):
    """
    Add comments to the TOML string under the appropriate sections.
    """
    for section, args in section_comments.items():
        section_header = f'["{section}"]'
        section_start = toml_str.find(section_header)
        if section_start == -1:
            continue  # Skip if the section is not found

        section_end = toml_str.find("\n[", section_start + len(section_header))
        section_end = section_end if section_end != -1 else len(toml_str)

        section_content_updated = toml_str[section_start:section_end]
        if descriptions[section]:
            section_content_updated = re.sub(
                f"({re.escape(section_header)})",
                rf"# {descriptions[section]}\n\1",
                section_content_updated,
            )
        for arg, desc in args.items():
            section_content_updated = re.sub(
                rf"({arg} = [^\n]+)",
                rf"# {desc}\n\1",
                section_content_updated,
                flags=re.MULTILINE,
            )

        # Replace the original section content with the updated one
        toml_str = (
            toml_str[:section_start] + section_content_updated + toml_str[section_end:]
        )

    return toml_str


def parse_help_text(help_text):
    """
    Parse the help text to extract argument names and descriptions.
    """
    description_pattern = re.compile(
        r"usage:.*?\n\n(.*?)(?=\n\noptional arguments:)", re.DOTALL
    )
    description = ""
    # Extract the description
    description_match = description_pattern.search(help_text)
    if description_match:
        description = description_match.group(
            1
        ).strip()  # Remove leading/trailing whitespace
    description = " ".join(description.split()).partition("(default:")[0].strip()
    # Regular expression to extract arguments and their help text
    help_text = help_text.split("show this help message and exit")[1]
    pattern = re.compile(
        r"--(\w+)(?:, -\w+)?[^\n]+\n\s+(.*?)(?=\(default:|\n  --|\n\n|\Z)", re.DOTALL
    )

    # Extract arguments and their descriptions
    extracted_args = pattern.findall(help_text)
    args_dict = {}
    for arg, desc in extracted_args:
        # Normalize the description text by removing leading/trailing whitespace and new lines
        # Also, exclude any default value information
        clean_desc = " ".join(desc.split()).partition("(default:")[0].strip()

        args_dict[arg] = clean_desc

    return description, args_dict


import concurrent.futures
from multiprocessing import Pool


def create_config(save_to):
    datasets = glob.glob(
        "src/generate_datasets/**/generate_dataset**.py", recursive=True
    )
    config = {}
    comments = {}
    descriptions = {}
    datasets_to_process = datasets
    with Pool(len(datasets_to_process)) as p:
        all_help_text = p.map(extract_help_text, datasets_to_process)

    for idx, dataset_path in tqdm(enumerate(datasets_to_process)):
        # help_text = extract_help_text(dataset_path)
        description, arg_help = parse_help_text(all_help_text[idx])
        dataset_name = "/".join(Path(dataset_path).parent.parts[-2:])
        module_name = ".".join(list(Path(dataset_path).parts)).strip(".py")

        module = importlib.import_module(module_name)
        if isinstance(module.DEFAULTS, list):
            for i, defaults in enumerate(module.DEFAULTS):
                section_name = f"{dataset_name}.{i}"
                config[section_name] = defaults
                comments[section_name] = arg_help
                descriptions[section_name] = description

        else:
            config[dataset_name] = module.DEFAULTS
            comments[dataset_name] = arg_help
            descriptions[dataset_name] = description
    # Convert the config to TOML format
    toml_str = toml.dumps(config)

    # In the create_config function
    toml_str_with_comments = add_comments_to_toml(toml_str, comments, descriptions)

    # Write the final TOML file
    with open(save_to, "w") as f:
        f.write(toml_str_with_comments)


def main(toml_all_path, toml_all_lite_path):
    create_config(toml_all_path)
    generate_lite(toml_all_path, toml_all_lite_path)


if __name__ == "__main__":
    main(
        toml_all_path="generate_all_datasets.toml",
        toml_all_lite_path="generate_all_datasets_lite.toml",
    )
