# Lecture Pattern Recognition 2019

This repository contains the exercise sheets as well as the code for the pattern recognition exercises. Solutions to the exercises will NOT be provided but only shown in the Monday exercise seminar slots.

## Exercise upload ##
Exercises need to be uploaded on **courses** at latest Tuesday after the exercise sessions. Only 1 version/group should be uploaded. Only the edited files should be uploaded and compressed in a .ZIP file.
Fill out the names of each group member in the group.txt file.

## Installation ##
To run the code, you will need to have a python 3.7.X version installed on your system.

We use the conda package environment to keep track of all packages. 

Download and install miniconda from https://docs.conda.io/en/latest/miniconda.html

If you want the full anaconda distribution, you can also install this instead of miniconda https://www.anaconda.com/distribution/

Both miniconda and anaconda includes the package and environment manager _conda_


Open a terminal and type `$ conda --version` to test if conda is installed.

With conda it is easy to have a separate virtual environment for each project. Run `$conda env list` to see the available environments (if you've just installed conda, only _base_ will be available). 

To install a new environment from the provided `.yml` file, run `$conda env create -f pr_conda_environment.yml`

To activate the `pattern recognition` environment, use `$conda activate pr`.

Run `which python` to find your virtual python environment directory (or `conda info -e` in the anaconda prompt). You will need to specify this directory to the IDE you are using.

To deactivate an active enrivonment, use `conda deactivate`.

Hint: Conda automatically activates the `base` environment when opening a terminal. If you do not want this, deactivate with: `$conda config --set auto_activate_base false`


For the programming IDE we recommend using PyCharm https://www.jetbrains.com/pycharm/download

To configure PyCharm to use the conda environment _pr_ you need to:

PyCharm -> Preferences -> Project: pattern-recognition-2019 -> Project Interpreter -> Add a new interpreter from an existing environment -> Insert the conda _pr_ environment path.

Students are able to get access to the full version by applying here: https://www.jetbrains.com/student/ but for the exercises here, the community version is sufficient.

To run the notebooks, execute `$jupyter notebook` in your _pattern-recognition-2019_ folder, which will start a jupyter server. 
## Contact ##

- Dennis Madsen <dennis.madsen@unibas.ch>
- Dana Rahbani <dana.rahbani@unibas.ch>
