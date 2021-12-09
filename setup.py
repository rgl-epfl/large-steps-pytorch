from setuptools import setup

setup(
    name='largesteps',
    version='0.1.1',
    description='Laplacian parameterization package for shape optimization with differentiable rendering',
    url='https://github.com/rgl-epfl/large-steps-pytorch',
    author='Baptiste Nicolet',
    author_email='baptiste.nicolet@epfl.ch',
    license='BSD',
    packages=['largesteps'],
    install_requires=['numpy',
                        'scipy',
                      ],
)
