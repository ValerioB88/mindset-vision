# Test Scripts and Samples Preview Workflow
## Overview
This folder contains components for managing and testing code in the project. It primarily includes test scripts and a system to preview sample outputs produced by the GitHub workflow.

## How It Works
The testing and preview workflow is integrated into the project's continuous integration (CI) system. Everytime as PR is merged, the _test.py script in the root directory is triggered. This is controlled by [this yml](.github/workflows/integration.yml)

Running test requires [pandoc](https://pandoc.org/installing.html)