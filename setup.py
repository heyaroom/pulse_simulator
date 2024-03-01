from setuptools import setup, find_packages

setup(
    name="pulse-simulator",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ]
)
