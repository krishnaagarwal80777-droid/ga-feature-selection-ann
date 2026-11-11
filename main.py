import argparse
import os

os.environ["PYTHONWARNINGS"] = (
    "ignore:Stochastic Optimizer:Warning:"
    "sklearn.neural_network._multilayer_perceptron"
)

from ga_fs.experiment import run_all, run_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    if args.noise:
        run_noise()
    else:
        run_all()
