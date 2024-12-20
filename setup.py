from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sick-bide",
    version="0.0.1",
    description="A Neural network layer able to express distributions over anything",
    author="Duon labs",
    author_email="contact@duonlabs.com",
    url="https://github.com/duonlabs/sick-bide",
    license="Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "sick_bide",
        "sick_bide.kernels",
        "sick_bide.kernels.bruteforce",
        "sick_bide.kernels.precompute",
    ],
    install_requires=[
        "torch",
        "triton",
        "numpy",
        "matplotlib",
        "lovely_tensors",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)