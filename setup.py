from setuptools import setup

setup(
    name='largesteps',
    version='0.1.0',
    description='Laplacian parameterization package for shape optimization with differentiable rendering',
    url='https://github.com/rgl-epfl/large-steps-pytorch',
    author='Baptiste Nicolet',
    author_email='baptiste.nicolet@epfl.ch',
    license='BSD 2-clause',
    packages=['largesteps'],
    install_requires=['numpy',
                        'cupy',
                        'scikit-sparse',
                        'scipy',
                      ],
)
