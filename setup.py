from setuptools import setup, find_packages

setup(
    name="data-science-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "black==22.12.0",
        "matplotlib==3.6.2",
        "pandas==1.5.2",
        "pytest==7.2.0",
    ],
)
