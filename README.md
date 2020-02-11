# pyemma-workshop
## Installation
We strongly recommend to install the latest `pyemma` release from the anaconda Python distribution.
### step 1: Miniconda
#### everyone
If you do not have miniconda or anaconda, please follow the instructions here for python 3.7:
https://conda.io/miniconda.html
In most cases for linux or mac users, a miniconda installation will suffice.

#### windows only
We recently experienced some installation problems with miniconda on windows and thus recommend anaconda for windows users.
- Get Anaconda from https://www.anaconda.com/distribution/#download-section
- Run the .exe downloaded file (link above)
- launch the Anaconda prompt
- update anaconda: `conda update -n base -c defaults conda`

### step 2: pyemma
Installation of all required software packages works by simply executing:

```bash
conda install -c conda-forge pyemma_tutorials
```

Please note that especially if you are already using conda, you might want to create a specific environment for the workshop. This is optional.

### step 3: activate some helpers
In order to activate some features of the notebooks that we will be using, please also run
```bash
jupyter nbextension enable toc2/main
jupyter nbextension enable exercise2/main
jupyter-nbextension enable nglview --py --sys-prefix
```

## Usage
### only on the first day
Please clone (download) this repository to get local access to the worksheets.

```bash
git clone https://github.com/markovmodel/pyemma_tutorials.git
```
Please remember *where* on your local hard disk you have written it!

### every morning:

#### activate environment (optional) 
Skip if you don't know what a conda environment is. Only if conda environment is used; name might differ.
``` bash
conda activate pyemma_tutorials
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

Your browser should pop up pointing to a list of notebooks. If it's the wrong browser, add for example `--browser=firefox`.
