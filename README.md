
<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center"><a href="https://bnicolet.com/publications/Nicolet2021Large.html">Large Steps in Inverse Rendering of Geometry</a></h1>

  <a href="https://bnicolet.com/publications/Nicolet2021Large.html">
    <img src="https://bnicolet.com/publications/images/Nicolet2021Large-teaser.jpg" alt="Logo" width="100%">
  </a>

  <p align="center">
    ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia), December 2021.
    <br />
    <a href="https://bnicolet.com/"><strong>Baptiste Nicolet</strong></a>
    ·
    <a href="https://www.cs.toronto.edu/~jacobson/"><strong>Alec Jacobson</strong></a>
    ·
    <a href="https://rgl.epfl.ch/people/wjakob"><strong>Wenzel Jakob</strong></a>
  </p>

  <p align="center">
    <a href='https://bnicolet.com/publications/Nicolet2021Large.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='https://bnicolet.com/publications/Nicolet2021Large.html' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
  </p>
</p>

<br />
<br />


<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
      <ul>
        <li><a href="#parameterization-package-installation">Parameterization package installation</a></li>
        <li><a href="#cloning-the-repository">Cloning the repository</a></li>
        <li><a href="#downloading-the-scenes">Downloading the scenes</a></li>
      </ul>
    </li>
    <li>
      <a href="#parameterization">Parameterization</a>
    </li>
    <li>
      <a href="#running-the-experiments">Running the experiments</a>
    </li>
    <li>
      <a href="#repository-structure">Repository structure</a>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgments">Acknowledgments</a>
    </li>
  </ol>
</details>
<br />
<br />

## Installation

This repository contains both the operators needed to use our parameterization
of vertex positions of meshes as well as the code for the experiments we show in
the paper.
### Parameterization package installation

If you are only interested in using our parameterization in an existing (PyTorch
based) pipeline, we have made it available to install via `pip`:

```bash
pip install largesteps
```

This will install the `largesteps` module. This only contains the
parameterization logic implemented as a PyTorch custom operator. See the
[tutorial](Tutorial.ipynb) for an example use case.

### Cloning the repository

Otherwise, if you want to reproduce the experiments from the paper, you can
clone this repo and install the module locally.

```bash
git clone --recursive git@github.com:rgl-epfl/large-steps-pytorch.git
cd large-steps-pytorch
pip install .
```

The experiments in this repository depend on PyTorch. Please follow instructions on
the PyTorch [website](https://pytorch.org/get-started/locally/) to install it.

To install `nvdiffrast` and the Botsch-Kobbelt remesher, which are provided as
submodules, please run the `setup_dependencies.sh` script.

`nvdiffrast` relies on the `cudatoolkit-dev` package to compile modules at runtime.
To install it with Anaconda:
```bash
conda install -c conda-forge cudatoolkit-dev
```

To install the other dependencies needed to run the experiments, also run:
```bash
pip install -r requirements.txt
```

:warning: On Linux, `nvdiffrast` requires using `g++` to compile some PyTorch
extensions, make sure this is your default compiler:

```bash
export CC=gcc && CXX=g++
```

Rendering the figures will also require installing
[blender](https://www.blender.org/download/). You can specify the name of the
blender executable you wish to use in `scripts/constants.py`

### Downloading the scenes

The scenes for the experiments can be downloaded
[here](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2021Large.zip).
Please extract the archive at the toplevel of this repository.

## Parameterization

In a nutshell, our parameterization can be obtained in just a few lines:

```python
# Given tensors v and f containing vertex positions and faces
from largesteps.geometry import laplacian_uniform, compute_matrix
from largesteps.parameterize import to_differential, from_differential

# Compute the system matrix
M = compute_matrix(v, f, lambda_=10)

# Parameterize
u = to_differential(M, v)
```

`compute_matrix` returns the parameterization matrix **M** = **I** + λ**L**.
This function takes another parameter, `alpha`, which leads to a slightly
different, but equivalent, formula for the matrix: **M** = (1-α)**I** + α**L**,
with α ∈ [0,1[. With this formula, the scale of the matrix **M** has the same
order of magnitude regardless of α.

```python
M = compute_matrix(L, alpha=0.9)
```

Then, vertex coordinates can be retrieved as:

```python
v = from_differential(u, M, method='Cholesky')
```

This will in practice perform a cache lookup for a solver associated to the
matrix **M** (and instantiate one if not found) and solve the linear system
**Mv** = **u**. Further calls to `from_differential` with the same
matrix will use the solver stored in the cache. Since this operation is
implemented as a differentiable PyTorch operation, there is nothing more to be
done to optimize this parameterization.

## Running the experiments

You can then run the experiments in the `figures` folder, in which each
subfolder corresponds to a figure in the paper, and contains two files:
- `generate_data.py`: contains the script to run the experiment and write the
  output to the directory specified in `scripts/constants.py`
- `figure.ipynb`: contains the script generating the figure, assuming
  `generate_data.py` has been run before and the output written to the directory
  specified in `scripts/constants.py`

We provide the scripts for the following figures:
- Fig. 1 -> `teaser`
- Fig. 3 -> `multiscale`
- Fig. 5 -> `remeshing`
- Fig. 6 -> `reg_fail`
- Fig. 7 -> `comparison`
- Fig. 8 -> `viewpoints`
- Fig. 9 -> `influence`

:warning: Several experiments are equal-time comparisons ran on a Linux Ryzen
3990X workstation with a TITAN RTX graphics card. In order to ensure
reproducibility, we have frozen the step counts for each method in these
experiments.

## Repository structure

The `largesteps` folder contains the parameterization module made available via
`pip`. It contains:
- `geometry.py`: contains the laplacian matrix computation.
- `optimize.py`: contains the `AdamUniform` optimizer implementation
- `parameterize.py`: contains the actual parameterization code, implemented as a
  `to_differential` and `from_differential` function.
- `solvers.py`: contains the Cholesky and conjugate gradients solvers used to
  convert parameterized coordinates back to vertex coordinates.

Other functions used for the experiments are included in the `scripts` folder:
- `blender_render.py`: utility script to render meshes inside blender
- `constants.py`: contains paths to different useful folders (scenes, remesher, etc.)
- `geometry.py`: utility geometry functions (normals computation, edge length, etc.)
- `io_ply.py`: PLY mesh file loading
- `load_xml.py`: XML scene file loading
- `main.py`: contains the main optimization function
- `preamble.py`: utility scipt to a import redundant modules for the figures
- `render.py`: contains the rendering logic, using `nvdiffrast`

## License
This code is provided under a 3-clause BSD license that can be found in the
LICENSE file. By using, distributing, or contributing to this project, you agree
to the terms and conditions of this license.


## Citation

If you use this code for academic research, please cite our method using the following BibTeX entry:

```bibtex
@article{Nicolet2021Large,
    author = "Nicolet, Baptiste and Jacobson, Alec and Jakob, Wenzel",
    title = "Large Steps in Inverse Rendering of Geometry",
    journal = "ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia)",
    volume = "40",
    number = "6",
    year = "2021",
    month = dec,
    doi = "10.1145/3478513.3480501",
    url = "https://rgl.epfl.ch/publications/Nicolet2021Large"
}
```
## Acknowledgments
The authors would like to thank Delio Vicini for early discussions about this
project, Silvia Sellán for sharing her remeshing implementation and help for the
figures, as well as Hsueh-Ti Derek Liu for his advice in making the figures.
Also, thanks to Miguel Crespo for making this README template.
