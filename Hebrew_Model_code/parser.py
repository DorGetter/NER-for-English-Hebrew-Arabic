import argparse
import numpy as np
import torch
import utils.parameters as param
from pathlib import Path


def set_seed(seed: int):
    if seed != None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class Parser(object):
    @staticmethod
    def train() -> argparse.Namespace:

        parser = argparse.ArgumentParser()

        general = parser.add_argument_group("general")
        general.add_argument("--seed", type=int, default=None, help="seed for reproducibility")
        general.add_argument("--name", type=str, default=param.todays_date_time, help="name of directory for product")

        dataset = parser.add_argument_group("dataset")
        dataset.add_argument("--train-file", type=Path, required=True, help="path to train file")
        dataset.add_argument(
            "--max-seq-len", type=int, default=param.max_sequence_length, help="maximal sequence length"
        )

        training = parser.add_argument_group("training")
        training.add_argument(
            "--finetune",
            action="store_true",
            default=False,
            help="set to finetune classifier rather than train entire model",
        )
        training.add_argument("--num-epochs", type=int, default=param.epochs, help="number of epochs to train")
        training.add_argument("--batch-size", type=int, default=param.batch_size, help="batch size")

        optimizer = parser.add_argument_group("optimizer")
        optimizer.add_argument("--learning-rate", type=float, default=param.learning_rate, help="learning rate")
        optimizer.add_argument(
            "--optimizer-eps", type=float, default=param.optimizer_tolerance, help="optimizer tolerance"
        )
        optimizer.add_argument(
            "--weight-decay-rate",
            type=float,
            default=param.optimizer_weight_decay_rate,
            help="optimizer weight decay rate",
        )
        optimizer.add_argument(
            "--max-grad-norm", type=float, default=param.maximal_gradients_norm, help="maximal gradients norm"
        )

        scheduler = parser.add_argument_group("scheduler")
        scheduler.add_argument(
            "--num-warmup-steps", type=int, default=param.scheduler_warmup_steps, help="scheduler warmup steps"
        )
        scheduler.add_argument(
            "--end-lr-factor",
            type=float,
            default=param.scheduler_end_lr_factor,
            help="scheduler final lr = `--learning-rate` * `--end-lr-factor`",
        )

        opts = parser.parse_args()
        set_seed(opts.seed)
        opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return opts

    @staticmethod
    def predict() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint directory")
        parser.add_argument("--sentence", type=str, default=None, help="sentence to apply NER")

        opts = parser.parse_args()
        opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return opts
