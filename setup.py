from setuptools import setup, find_packages

setup(
    name="data-science-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "black==22.3.0",
        "matplotlib==3.5.2",
        "pandas==1.2.4",
    ],
)
