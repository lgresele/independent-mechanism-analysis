# Idependent mechanism analysis


## Installation

In order to install the package and all dependencies, run

```
pip install --upgrade git+https://github.com/lgresele/ica_and_icm.git
```
Note a different version of `jaxlib` might have to be installed to enable GPU acceleration.


## Reproduce experiments

The data and the models used to produce Figure 4 (top) can be downloaded
[here](https://drive.google.com/drive/folders/1js-doh_b1pWBBg-VtQd1gVjkBopRUXYm?usp=sharing).

The experiments can be reproduced by running the respective script stored in the `experiments` folder and passing it the respective configuration file given in the `config` folder. If several seeded runs where done, the integers from 0 to m-1 are taken as seeds if m is the number of runs we did.

The figures where produced using the notebooks stored in the `figures` folder.
