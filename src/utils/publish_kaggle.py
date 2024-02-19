import os
import subprocess
import json
import sty


def generate_slug(name):
    slug = "".join([c if c.isalnum() or c == "-" else "-" for c in name])
    return slug


def publish(data_type="full"):
    if data_type == "lite":
        folder_name = "data_lite"
    else:
        folder_name = "data"

    title = "MindSet: Vision" + (" (lite)" if data_type == "lite" else "")
    slug = "mindset" + ("-lite" if data_type == "lite" else "")

    print(sty.fg.red + f"Generating {title}" + sty.rs.fg)

    metadata = {
        "title": title,
        "id": f"Valerio1988/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }

    metadata_path = os.path.join(f"{folder_name}", "dataset-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    subprocess.call(
        [
            "kaggle",
            "datasets",
            "version",  # should be "create" if first time you upload it
            "-p",
            f"{folder_name}",
            "-r",
            "zip",  # Upload folder as a zip
            "-m",
            "Updating {title} dataset",
            "-d",
        ]
    )


if __name__ == "__main__":
    publish(data_type="full")
    # publish(data_type="lite")
