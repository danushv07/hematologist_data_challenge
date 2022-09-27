# Hematology_data_challenge

## Description
This repo is for the **Help a hematologist out** data challenge. The solution represented here won the $`2^{nd}`$ place in this challenge. The dataset and the information is available [here](https://helmholtz-data-challenges.de/web/challenges/challenge-page/93/overview)

The challenge is to develop models/techniques for domain adaptation. As a part of the ongoing study, the equivariant models indicated better generalization than conventional models. Through this challeneg we intend to test the hypothesis for domain adaptation problems.

The $`D_{8}`$ equivariant model indicated the best performance

## Usage
1. Create a virtual env. and install the dependencies using the file ```requirements.txt```
Pytorch and e2cnn has been used to design the models. When using ```A100``` GPUs certain errors/issues arise with pytorch and [e2cnn](https://github.com/QUVA-Lab/e2cnn). First install pytorch from [here](https://pytorch.org/)
```bash
python3 -m venv hemat
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
Then install the required dependencies using
```bash
python3 -m pip install -r requirements.txt
```
- Train the model
```bash
cd src/
python train.py --n_rotation 8 --n_filters 32 --flip True --num_class 11 --save_path path/to/save/model --model_type "res" --epoch 10 --ace_path path/to/acevedo --mat_path path/to/matek 
```
- To run inference
```bash
python eval.py --n_rotation 8 --n_filters 32 --flip True --num_class 11 --file_path path/to/saved_model --save_path path/to/save_submission 
```
2. For a fine grained analysis of the models the ```train_book``` can be utilized. The book contains self explanatory code which can be used to train and validate the models. The confusion matrix for acevedo and matek datasets can also be visualized.

- Create the environment using
```
conda env create -f environment.yml
```

## Support
For questions kindly open an issue.

## Acknowledgement
This work is supported by the Helmholtz Association Initiative and Networking Fund under the Helmholtz AI platform grant.
