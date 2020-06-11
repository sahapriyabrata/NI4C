# Neural Identification for Control

## Installation

Compatible with Python 3.5 and Pytorch 1.1.0

1. Create a virtual environment by `python3 -m venv env`
2. Source the virtual environment by `source env/bin/activate`
3. Install requirements by `pip install -r ./requirements.txt`

## Usage

Go to nLinkPendulum directory: `cd nLinkPendulum`

# Dataset generation

To generate dataset, run
`python dataGen --set <train/val>

# Training

To train NN_g, run 
`python train_NNg.py`

To train NN_P and NN_pi, run
`python train_NNpiP.py`
