"""
Setup script for Monetary Policy RL package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="monetary_policy_rl",
    version="1.0.0",
    author="Hacene Tchoketch-Kebir",
    author_email="hassentchoketch@gmail.com",
    description="Optimal Monetary Policy using Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hassentchoketch/optimal_monetary_policy_with_rl-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
)