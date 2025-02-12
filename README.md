# wenglor ONNX Model Export Example

## Overview

This repository contains a [Jupyter notebook](notebooks/model-export-example.ipynb) and utility scripts for training, quantizing and exporting a PyTorch model to
a `.u3o` file, the format accepted by uniVision Module Image ONNX. It is a zip archive with an ONNX model file and a metadata in `YAML` format.

### Directory Structure

```shell
/notebooks/
    ├── model-export-example.ipynb         # End-to-end example notebook
    ├── utils/                             # Folder containing utility scripts 
    ├── images/                            # Folder containing example images for model training
Dockerfile                                 # Docker configuration
docker-compose.yml                         # Docker Compose configuration
requirements.txt                           # List of Python dependencies
```

## Prerequisites

To follow the example, make sure you have the following installed:

- [Docker](https://docs.docker.com/engine/install/) (recommended)
- [Python 3.8+](https://www.python.org/downloads/) (optional, for system Python setup)
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) (optional, for venv setup)
- [Jupyter Notebook](https://jupyter.org/install) (if running outside Docker)

## Installation and Setup

### Option 1: Using Docker Compose

This is the simplest method to get started.

1. Build and start the services.

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

### Option 2: Using Virtual Environment (venv)

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

### Option 1

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
