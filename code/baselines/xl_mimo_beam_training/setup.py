"""Setup script for xl_mimo_beam_training package."""

from setuptools import find_packages, setup

setup(
    name="xl_mimo_beam_training",
    version="1.0.0",
    description="Near-Field Beam Training for XL-MIMO Using Deep Learning",
    author="J. Nie, Y. Cui et al.",
    python_requires=">=3.10,<3.13",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.1",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "matplotlib==3.10.9",
        "scikit-learn==1.5.2",
        "PyYAML==6.0.2",
    ],
    extras_require={
        "dev": ["pytest==8.3.5"],
    },
)
