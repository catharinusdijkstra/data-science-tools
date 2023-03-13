# README #

## Introduction

Welcome to my **data-science-tools** repository. In this repository, you find the
**cdtools** package with code useful for data science analysis.

The codebase in this repository can be directly imported in your own Python code for
data science analysis, or it can be used to setup a Python environment
data-science-tools from which all functionalities in this repository can be run locally
and unit tests can be performed.

## Requirements

1. An Integrated Development Environment (IDE) such as for example
[Visual Studio Code](https://code.visualstudio.com/).
2. [conda](https://docs.conda.io/projects/conda/en/stable/).

## Installation of this repository

To use this repostory in your own Python code you first need to install it using
[pip](https://pypi.org/project/pip/). This can be done from the command line into two
ways.

The first way is simply by running the following command:

```
pip install git+https://github.com/catharinusdijkstra/data-science-tools.git
```

The second way is to create a new **requirements.txt** file and include the line
git+<https://github.com/catharinusdijkstra/data-science-tools.git>, or if you already
have a **requirements.txt** file with any other packages you want to install, simply
add this line to this already existing file. Then run the following command:

```
pip install -r requirements.txt
```

## Setup the Python environment data-science-tools locally (optional)

In order to setup the Python environment data-science-tools locally, run the following
commands from the root folder of this repository:

```
conda env update --file environment.yml
conda activate data-science-tools
```

This will create and activate the Python environment data-science-tools from which
all functionalities in this repository can be run locally and unit tests can be
performed.

Note that this step is optional. This step is not needed for direct use of the cdtools
package in your own Python code. All that is required for that are the installation
instructions from the previous step.

## How to use this repository in your own code

After installation, open a Python program and/or Jupyter notebook and import the package
cdtools and/or its sub-packages and modules using the Python commands **import** and
**from**. For example:

```
import cdtools
from cdtools import *
```

You can now use the functionalities within the **cdtools** package for data science
analysis.
