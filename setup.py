#!/usr/bin/env python3

from setuptools import setup,find_packages,find_namespace_packages
from mb_tensorflow.utils.version import version

setup(
    name="mb_tensorflow",
    version=version,
    description="Tensorflow functions package",
    author=["Malav Bateriwala"],
    packages=find_namespace_packages(include=["mb_tensorflow.*"]),
    #packages=find_packages(),
    scripts=[],
    install_requires=[
        "numpy",
        "pandas",],
    python_requires='>=3.8',)
