from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GPU-Accelerated-Climate-and-Water-Level-Forecasting-System-using-Deep-Learning",
    version="0.1.0",
    author="Jayesh, Mahadi",
    packages=find_packages(),
    install_requires=requirements,
    description="GPU-accelerated system for forecasting climate variables and water levels using advanced deep learning models",
)