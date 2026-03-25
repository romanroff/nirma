# NIRMA

## Quickstart

- `py -3.12 -m venv .venv` - create the virtual environment
- `.venv\Scripts\Activate.ps1` - activate it in PowerShell
- `python -m pip install --upgrade pip` - upgrade pip inside the venv
- `pip install -r requirements.txt` - install runtime and notebook dependencies
- `python -m ipykernel install --user --name nirma-venv --display-name .venv` - register the Jupyter kernel used by the notebooks
