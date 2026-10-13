"""Train transformer_sac."""
from utils.train_baselines import parse_args, train

if __name__ == "__main__":
    train(parse_args("transformer_sac"))
