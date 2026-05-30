from __future__ import annotations

import argparse

from src.train import train
from src.predict import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=["train", "predict", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ["train", "all"]:
        train(args.config)
    if args.stage in ["predict", "all"]:
        predict(args.config)


if __name__ == "__main__":
    main()
