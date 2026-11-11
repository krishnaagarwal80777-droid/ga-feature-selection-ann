import argparse
import os

os.environ["PYTHONWARNINGS"] = (
    "ignore:Stochastic Optimizer:Warning:"
    "sklearn.neural_network._multilayer_perceptron"
)

from ga_fs.experiment import (
    rerun_baselines,
    run_all,
    run_noise,
    run_sensitivity,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--noise", action="store_true")
    group.add_argument("--sensitivity", action="store_true")
    group.add_argument("--rerun-baselines", action="store_true")
    args = parser.parse_args()
    if args.noise:
        run_noise()
    elif args.sensitivity:
        run_sensitivity()
    elif args.rerun_baselines:
        rerun_baselines()
    else:
        run_all()
