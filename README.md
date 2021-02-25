# pyemma-workshop

## Installation
We strongly recommend to install the latest `pyemma` and `deeptime` release from the anaconda Python distribution.

### step 1: Miniconda
If you do not have miniconda or anaconda, please follow the instructions here for Python 3.8: https://conda.io/miniconda.html

We recommend to create a separate environment for the workshop, especially if you already have a anaconda/miniconda installation:
```
# these steps are optional but recommended
conda create -n workshop
conda activate workshop
conda install python=3.8

# this is not optional
conda config --env --add channels conda-forge
```

---
**NOTE**

For Windows users it makes sense to also install GIT if it is not already available on the system: ``conda install git``

---

### step 2: pyemma and deeptime
Installation of all required software packages works by simply executing:

```bash
conda install pyemma_tutorials
```

### step 3: activate some helpers
In order to activate some features of the notebooks that we will be using, please also run
```bash
jupyter nbextension enable toc2/main
jupyter nbextension enable exercise2/main
jupyter-nbextension enable nglview --py --sys-prefix
```

## Sanity check

You can check whether you installed the correct versions by calling
```
conda list
```

PyEMMA should show up with version `2.5.8` and deeptime with version `0.2.5`.

## Usage
### only on the first day
Please clone (download) this repository to get local access to the worksheets.

```bash
git clone https://github.com/markovmodel/pyemma-workshop.git
```
Please remember *where* on your local hard disk you have written it!

### every morning:

#### activate environment (optional) 
Skip if you don't know what a conda environment is. Only if conda environment is used; name might differ.
``` bash
conda activate workshop
```

#### navigate to the right folder
Please navigate to the folder that you cloned from our github page.
```bash
cd path/to/pyemma-workshop/notebooks
```

#### start the jupyter notebook server
This command will start the notebook server:
```bash
jupyter notebook
```

Your browser should pop up pointing to a list of notebooks. If it's the wrong browser, add for example `--browser=firefox` or copy and paste the URL into the browser of your choice.

### getting updates
Once you have a local clone of this repository, you can easily obtain updates with `git pull`. 
We'll let you know once we have published anything new.
If you work directly in the notebooks that we provide, you might have to use the sequence (`git pull` will raise an error):
```bash
git stash
git pull
git stash pop
```
