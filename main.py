import os

os.environ["PYTHONWARNINGS"] = (
    "ignore:Stochastic Optimizer:Warning:"
    "sklearn.neural_network._multilayer_perceptron"
)

from ga_fs.experiment import run_all


if __name__ == "__main__":
    run_all()
