from setuptools import setup, find_packages

setup(
    name="data-science-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "black",
        "imbalanced-learn",
        "ipykernel",
        "ipywidgets",
        "lightgbm",
        "lime",
        "matplotlib",
        "mlxtend",
        "openpyxl",
        "pandas",
        "pandas-profiling",
        "pytest",
        "seaborn",
        "scikit-learn",
        "treeinterpreter",
        "xlrd",
        "yellowbrick",
    ],
)
