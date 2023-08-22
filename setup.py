from setuptools import setup

# read the contents of your README file (https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/)
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

setup(
    name='largesteps',
    version='0.2.2',
    description='Laplacian parameterization package for shape optimization with differentiable rendering',
    url='https://github.com/rgl-epfl/large-steps-pytorch',
    author='Baptiste Nicolet',
    author_email='baptiste.nicolet@epfl.ch',
    license='BSD',
    packages=['largesteps'],
    install_requires=['numpy',
                      'scipy',
                      'cholespy'
                      ],
    long_description=readme,
    long_description_content_type="text/markdown"
)
