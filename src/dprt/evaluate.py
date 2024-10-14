import argparse

from dprt.datasets import init
from dprt.datasets import load
from dprt.evaluation import evaluate
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed


def main(src: str, cfg: str, checkpoint: str, dst: str):
    """ Data preparation for subsequent model training or evaluation.

    Arguments:
        scr: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        dst: Destination directory to save the processed dataset files.
    """
    # Load dataset configuration
    config = load_config(cfg)

    # Set global random seed
    set_seed(config['computing']['seed'])

    # Initialize test dataset
    test_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)

    # Load test dataset
    test_loader = load(test_dataset, config=config)

    # Evaluate model at checkpoint
    evaluate(config)(checkpoint, test_loader, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/data/kradar/processed',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/app/config/kradar.json',
                        help="Path to the configuration file.")
    parser.add_argument('--checkpoint', type=str, default='/app/log/',
                        help="Path to save the evaluation log.")
    parser.add_argument('--dst', type=str, default='/app/log',
                        help="Path to save the processed dataset.")
    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, checkpoint=args.checkpoint, dst=args.dst)
