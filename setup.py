from setuptools import setup, find_packages

setup(
    name="bilm",
    version="2.0.0a1",
    description="Experimental byte-native language model for CPU continual learning.",
    author="UnikAI Lab",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.57.0",
    ],
    extras_require={"dev": ["pytest>=7.0", "pytest-cov>=4.0"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
