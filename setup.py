from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="saluslux",
    version="0.1.0",
    description="A Python package for simulating and optimizing urban lighting using photometric data",
    author="Korawich Kavee",
    author_email="kkavee@andrew.cmu.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

long_description=description,
long_description_content_type="text/markdown"