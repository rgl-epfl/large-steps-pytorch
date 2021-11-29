
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

## How to use this repo?

This repository contains both the operators needed to use our parameterization
of vertex positions of meshes as well as the code for the experiments we show in
the paper.

If you are only interested in using our parameterization in an existing (PyTorch
based) pipeline, you can simply install it with:

```bash
pip install largesteps
```

Otherwise, you can clone this repo and install the module locally:
```bash
git clone --recursive git@github.com:rgl-epfl/large-steps-pytorch.git
cd large-steps-pytorch
pip install .
```

This will install the `largesteps` module. This only
contains the parameterization logic implemented as a PyTorch custom op:
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

See the [tutorial](Tutorial.ipynb) for an example use case.

You can also run the experiments in the `figures` folder, in which each subfolder corresponds to a figure in the paper, and
contains two files:
- `generate_data.py`: contatins the script to run the experiment and write the
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

## License
This code is provided under a BSD-style license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

## Acknowledgments
The authors would like to thank Delio Vicini for early discussions about this
project, Silvia Sellán for sharing her remeshing implementation and help for the
figures, as well as Hsueh-Ti Derek Liu for his advice in making the figures.
Also, thanks to Miguel Crespo for making this README template.
