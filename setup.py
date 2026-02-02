#!/usr/bin/env python3

import os
import re
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


VERBOSE_SCRIPT = True
for arg in sys.argv:
    if arg == "-q" or arg == "--quiet":
        VERBOSE_SCRIPT = False


def report(*args):
    if VERBOSE_SCRIPT:
        print(*args)
    else:
        pass


TUTORIALS_REQUIRES = ["torchtext", "torchvision"]

# TODO: review if all of them are still needed
TEST_REQUIRES = [
    "pytest",
    "pytest-cov",
    "parameterized",
    "black==25.11.0",
    "flake8",
    "pyre-check-nightly==0.0.101750936314",
    "usort==1.1.0",
    "ufmt",
]

# captum may have some functionality that requires these packages, but they are
# not required for the core functionality of captum.
# These packages should be lazily imported.
OPTIONAL_REQUIRES = [
    "openai",  # remote
    "scikit-learn",
    "annoy",  # influence
]

DEV_REQUIRES = (
    TEST_REQUIRES
    + OPTIONAL_REQUIRES
    + [
        "sphinx<8.2.0",
        "sphinx-autodoc-typehints",
        "sphinxcontrib-katex",
    ]
)

# get version string from module
with open(os.path.join(os.path.dirname(__file__), "captum/__init__.py"), "r") as f:
    version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    assert version_match is not None, "Unable to find version string."
    version = version_match.group(1)
    report("-- Building version " + version)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setup(
        name="captum",
        version=version,
        description="Model interpretability for PyTorch",
        author="PyTorch Team",
        license="BSD-3",
        url="https://captum.ai",
        project_urls={
            "Documentation": "https://captum.ai",
            "Source": "https://github.com/pytorch/captum",
            "conda": "https://anaconda.org/pytorch/captum",
        },
        keywords=[
            "Model Interpretability",
            "Model Understanding",
            "Model Interpretability",
            "Model Understanding",
            "Feature Importance",
            "Neuron Importance",
            "Data Attribution",
            "Explainable AI",
            "PyTorch",
        ],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering",
        ],
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=">={required_major}.{required_minor}".format(
            required_minor=REQUIRED_MINOR,
            required_major=REQUIRED_MAJOR,
        ),
        install_requires=[
            "matplotlib",
            "numpy",
            "packaging",
            "torch>=2.3",
            "tqdm",
        ],
        packages=find_packages(exclude=("tests", "tests.*")),
        extras_require={
            "dev": DEV_REQUIRES,
            "test": TEST_REQUIRES,
            "tutorials": TUTORIALS_REQUIRES,
        },
    )
