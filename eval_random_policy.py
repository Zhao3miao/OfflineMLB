"""Evaluate the uniform-random action baseline on map or MRO targets."""
import argparse

from utils.experiment_utils import add_common_eval_args, collect_targets, evaluate_baseline, write_json


def evaluate(args):
    targets = collect_targets(args)
    result = evaluate_baseline(targets, "random", args.seed, args.max_steps)
    write_json(args.output, result)
    print("mean_load_std=%.6f episodes=%d output=%s" %
          (result["mean_load_std"], result["num_episodes"], args.output), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the uniform-random baseline.")
    add_common_eval_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
