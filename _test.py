"""
Script for regression test / revisit doc string later
"""

from src.utils.generate_default_pars_toml_file import create_config, generate_lite
from src.generate_datasets_from_toml import generate_datasets_from_toml_file
from pathlib import Path
import toml
import inspect

import shutil
import os
from collections import defaultdict
import base64
from nbformat import v4 as nbf
from nbconvert import PDFExporter
from src.utils.misc import modify_toml


def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_bytes = base64.b64encode(img_file.read())
    return encoded_bytes.decode("utf-8")


def build_dataset_structure(path, dataset_structure):
    # collect first 9 images from a dataset for displaying
    for root, dirs, files in sorted(os.walk(path)):
        image_files = [
            Path(root) / file for file in sorted(files) if file.endswith(".png")
        ][:9]
        toml_file = [
            Path(root) / file for file in sorted(files) if file.endswith(".toml")
        ]
        if toml_file:
            d = dataset_structure
            path_parts = Path(root).relative_to(path).parts
            for part in path_parts:
                d = d.setdefault(part, defaultdict(dict))
            d["config"] = toml.load(toml_file) if toml_file else None
        if image_files:
            d = dataset_structure
            path_parts = Path(root).relative_to(path).parts
            for part in path_parts:
                d = d.setdefault(part, defaultdict(dict))
            d["images"] = image_files


def generate_headers(nb, data, level=1):
    if "config" in data:
        config = data.pop("config")
        config_str = toml.dumps(config)
        nb.cells.append(nbf.new_markdown_cell(f"```toml\n{config_str}\n```"))

    for key, value in sorted(data.items()):
        if key != "images":
            nb.cells.append(nbf.new_markdown_cell(f"{'#' * level} {key}"))

            generate_headers(nb, value, level + 1)
        else:
            nb.cells.append(nbf.new_markdown_cell(create_table_markdown(value)))


def create_table_markdown(images):
    table_markdown = "<table><tr>"
    for index, image_path in enumerate(images):
        base64_data = encode_image_base64(image_path)
        if base64_data:
            table_markdown += f"<td><img src='data:image/png;base64,{base64_data}' alt='{image_path.name}'></td>"
            if (index + 1) % 3 == 0 or (index + 1) == len(images):
                table_markdown += "</tr><tr>"
    table_markdown += "</tr></table>"
    return table_markdown


def check_up_config(project_name: str):
    """using project name to look up the used config"""
    toml_path = Path("tests", "tomls", f"{project_name}.toml")
    assert toml_path.parent.exists(), "Tomls folder not found"
    if toml_path.exists():
        with open(toml_path, "r") as f:
            toml_config = toml.load(f)
            return toml_config
    return ""


from nbconvert import HTMLExporter


def generate_summary(dataset_structure, save_to):
    # Create a new notebook
    nb = nbf.new_notebook()
    generate_headers(nb, dataset_structure)

    # Save the notebook file
    with open(save_to, "w") as f:
        f.write(nbf.writes(nb))

    # Convert and save the notebook as HTML
    html_exporter = HTMLExporter()
    html_data, _ = html_exporter.from_notebook_node(nb)
    save_to_html = str(save_to).replace(".ipynb", ".html")
    with open(save_to_html, "w") as f:
        f.write(html_data)


import src.utils.generate_default_pars_toml_file
import sty


def test_generate_toml():
    source_toml = "tests/generate_all_datasets.toml"
    source_toml_lite = "tests/generate_all_datasets_list.toml"
    print("*** GENERATING DEFAULT TOML FILES: tests/generate_all_datasets.toml ***")
    create_config(source_toml)
    generate_lite(source_toml, source_toml_lite)

    print(sty.fg.blue + "*** DONE *** " + sty.rs.fg)
    print(
        sty.fg.blue + "*** GENERATING SMALL SAMPLES, BLACK BACKGROUND ***" + sty.rs.fg
    )

    name_test = "small_black_bg"
    with open(source_toml, "r") as file:
        toml_lines = file.readlines()

    toml_lines = modify_toml(
        toml_lines,
        modified_key_starts_with="num_samples",
        modify_value_fun=lambda h, x: 5,
    )

    generate_test_dataset_and_notebook_from_toml_file(
        toml_lines, source_toml, name_test
    )

    print(
        sty.fg.blue + "*** GENERATING SMALL SAMPLES, RANDOM BACKGROUND ***" + sty.rs.fg
    )
    name_test = "small_random_bg"
    with open(source_toml, "r") as file:
        toml_lines = file.readlines()

    toml_lines = modify_toml(
        toml_lines,
        modified_key_starts_with="num_samples",
        modify_value_fun=lambda h, x: 5,
    )

    toml_lines = modify_toml(
        toml_lines,
        modified_key_starts_with="background_color",
        modify_value_fun=lambda h, x: '"rnd-uniform"',
    )
    print(toml_lines)
    generate_test_dataset_and_notebook_from_toml_file(
        toml_lines, source_toml, name_test
    )


def generate_test_dataset_and_notebook_from_toml_file(
    toml_lines, source_toml, name_test
):
    toml_lines = modify_toml(
        toml_lines,
        modified_key_starts_with="output_folder",
        modify_value_fun=lambda h, x: f'"tests/{name_test}/data/{h.strip(chr(34))}"',
    )

    name_toml_file = Path(os.path.splitext(source_toml)[0] + "_" + f"{name_test}.toml")
    with open(name_toml_file, "w") as file:
        file.writelines(toml_lines)
    save_path = Path("tests") / name_test
    if Path(save_path).exists():
        shutil.rmtree(save_path)

    generate_datasets_from_toml_file(name_toml_file)
    dataset_structure = defaultdict(dict)
    build_dataset_structure(save_path / "data", dataset_structure)
    generate_summary(dataset_structure, save_path / "summary.ipynb")


if __name__ == "__main__":
    test_generate_toml()
