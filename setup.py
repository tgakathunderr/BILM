from setuptools import setup, find_packages

setup(
    name="bilm",
    version="1.0.0",
    description="Biologically Inspired Language Model — a continuously-learning, CPU-native alternative to transformers.",
    author="UnikAI Lab",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.57.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
