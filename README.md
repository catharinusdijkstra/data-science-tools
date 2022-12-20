# README #

## Introduction

Welcome to my **data-science-tools** repository. In this repository, you find the
**cdtools** package with code useful for data science analysis.

The codebase in this repository is intended to be imported in your personal local data
science project repository, where it can be used for data science analysis.

## Requirements

1. An Integrated Development Environment (IDE) such as for example
[Visual Studio Code](https://code.visualstudio.com/).
2. [conda](https://docs.conda.io/projects/conda/en/stable/).

## Installation of this repository locally

Run the following commands from the root folder of this repository

```
conda env update --file environment.yml
conda activate data-science-tools
```

This will create and activate the Python environment **data-science-tools** from which
all functionalities in this repository can be run locally and unit tests can be
performed.

## Installation of this repository for use in your local data science repository

First, add the data-science-tools repository to the same folder as your local data
science project repository. For example, if you have the following folder to store
your repositories

```/home/my_user_name/my_repositories/```

and your local data science project repository is located in

```/home/my_user_name/my_repositories/your_local_data_science_project_repository```

then store the code of the data-science-tools repository in

```/home/my_user_name/my_repositories/data-science-tools```

Second, in your local data science project repository

```/home/my_user_name/my_repositories/your_local_data_science_project_repository```

create the file environment.yml file and include the following lines of code in it:

```
name: name_of_your_local_data_science_repository
channels:
  - conda-forge
dependencies:
  - python=python_version_number
  - pip
  - pip:
      - ipykernel
      - ../data-science-tools
```

where you replace **name_of_your_local_data_science_repository** with the name
appropriate for your local data sciene repository and **python_version_number** with the
python version number you are using in your local data sciene repository.

Third, run the following commands from the root folder of your local data sciene
repository

```/home/my_user_name/my_repositories/your_local_data_science_project_repository```

```
conda env update --file environment.yml
conda activate name_of_your_local_data_science_repository
```

Finally, open a Python program and/or Jupyter notebook and import the packages
**cdtools** and/or its sub-packages and modules using the Python commands **import** and
**from**. For example:

```
import cdtools
from cdtools import *
```
