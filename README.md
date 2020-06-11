# Neural Identification for Control

## Installation

Compatible with Python 3.5 and Pytorch 1.1.0

1. Create a virtual environment by `python3 -m venv env`
2. Source the virtual environment by `source env/bin/activate`
3. Install requirements by `pip install -r ./requirements.txt`

## Usage

Go to **nLinkPendulum** directory: `cd nLinkPendulum`

### Dataset generation

To generate dataset, run:  
`python dataGen.py --set <train/val> --savepath <path to save dataset>`

### Training

To train NN_g, run:  
`python train_NNg.py --dataset <path to dataset> --savepath <path to save models>`

To train NN_P and NN_pi, run:   
`python train_NNpiP.py --NNg <path to trained NN_g> --dataset <path to dataset> --savepath <path to save models>`

### Evaluation

Pretrained models for 2-link pendulum are given in **saved_models** directory  

For a demo, run:  
`python test.py --modelpath <path to a trained model NI4C> --savepath <path to save result>` 

To verify a learned control law, run:  
`python controller_verification.py --modelpath <path to a trained model NI4C> --init <grid/random> --savepath <path to save result>`
