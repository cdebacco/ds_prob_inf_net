# Development enviroment

Instructions on how to replicate our research either downloading our package by using the Virtualenv setup used for developing the package.

If you are familiar with Virtualenv, you could use the Virtualenv provided in this repository. Use this setup if you want to modify anything in the package.

1. [Clone the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository) to a directory in your machine
2. [Set up Python, Pip and Virtualenv](http://timsherratt.org/digital-heritage-handbook/docs/python-pip-virtualenv/)

3. Update pip

```{console}
python -m pip install --upgrade pip
```

3. Open the terminal and build the project:
```{console}
cd <package_name>
<!-- python -m virtualenv venv -->
python3.8 -m venv venv
source venv/bin/activate # if on Windows: .\venv\Scripts\activate
```
4. Install the package and deps:
```{console}
pip install -r requirements.txt
pip install -e .
```
5. Create a JupyterLab instance by running the following command:
```{console}
jupyter lab
```

### Python 3.12
If you have Python 3.12 you may have to run:
```
python3 -m pip install setuptools
```
See [here](https://stackoverflow.com/questions/77233855/why-did-i-get-an-error-modulenotfounderror-no-module-named-distutils)

It still may create conflicts, so it's better to use Python < 3.12, e.g. Python 3.10.11