from setuptools import setup, find_packages

setup(
    name="data-science-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "black",
        "matplotlib",
        "pandas",
        "pytest",
    ],
)
