# wenglor ONNX Model Export Example

## Overview

This repository contains a [Jupyter notebook](notebooks/model-export-example.ipynb) and utility scripts for training, quantizing and exporting a PyTorch model to
a `.u3o` file, the format accepted by uniVision Module Image ONNX. It is a zip archive with an ONNX model file and a metadata in `YAML` format.

*DISCLAIMER:* Note that all examples are provided to demonstrate the complete model creation workflow with a focus on the exported `.u3o` file.
Training code and example images are placeholders given for purely for illustration, you will need to replace them with your actual training pipeline
and dataset.

### Directory Structure

```shell
├── data
│   ├── images                             # Folder containing example images for model training
│   └── model                              # Folder where resulting models will be stored
├── notebooks
│   ├── data
│   ├── utils                              # Folder containing utility scripts
│   └── model-export-example.ipynb         # End-to-end example notebook
├── Dockerfile                             # Docker configuration
├── docker-compose.yml                     # Docker Compose configuration
└── requirements.txt                       # List of Python dependencies                
```

(recreate using `tree -L 2 --dirsfirst`)

## Prerequisites

To follow the example have two options: if you have Docker installed, you can build an environment that has all requirements inside the Docker image. Alternatively, you can set up your environment on your  Linux, Mac, or Windows host

### Option 1: Installation using Docker (recommended)

FYI we use docker engine within a WSL2 environment on Windows. We also tested this with Ubuntu 22.04, newer versions should also work.

- [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Docker](https://docs.docker.com/engine/install/)

### Option 2: Installation without Docker

- [Python 3.8+](https://www.python.org/downloads/) (optional, for system Python setup without Docker)
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) (optional, for venv setup without Docker)
- [Jupyter Notebook](https://jupyter.org/install) (if running outside Docker)

## Installation and Setup

### Option 1: Using Docker Compose (recommended)

This is the simplest method to get started.

1. Build and start the services.

   ```bash
   docker compose up --build
   ```

   Internal user will default to the user with id 1000, which is the first user created on Ubuntu and will work in most cases.
   Using the same user allows you to change the notebook file on the host from within the container.
   If you have a different user, you can provide USER_ID and GROUP_ID parameters and the
   container will run jupyter lab using the provided user and group.

   ```bash
   export USER_ID=$(id -u)
   export GROUP_ID=$(id -g)
   docker compose up --build
   ```

2. Access the Jupyter Notebook. After running the above command, navigate to the printed Jupyter URL in your browser (usually `http://localhost:8888`). When the Docker container starts, a token will be printed in the terminal output. Look for a URL containing `?token=`, followed by a string of characters. Copy this token from the terminal and append it to the Jupyter URL in your browser, e.g. `http://localhost:8888/lab?token=...`.

3. Stop the service:

   ```bash
   docker compose down
   ```

### Option 2: Using Virtual Environment (venv) - when not using Docker

1. Create and activate a virtual environment.

   On macOS/Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Navigate to the `notebooks/` folder and open `.ipynb` file.

5. Deactivate the virtual environment after you're done:

   ```bash
   deactivate
   ```

### Option 3: Using System Python

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Navigate to the `notebooks/` folder and open `.ipynb` file.

## Troubleshooting

### Permission problems with Docker (Option 1)

`PermissionError: [Errno 13] Permission denied: '/home/{USER}/.local/share'`

   1. Check Ownership:
      Verify that you own the directories in question

      ```bash
      ls -ld ~/.local ~/.local/share
      ```

      The output should show that you are the owner. If not, you need to change ownership:

      ```bash
      sudo chown -R $(whoami):$(whoami) ~/.local
      ```

   2. Permission for Directories:
      Sets read, write, and execute permissions for the user, and read and execute for the group and others

      ```bash
      chmod -R 755 ~/.local
      ```
