import logging
import os

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def get_eval_logger(exp_dir):
    log_path = os.path.join(exp_dir, "evaluation.log")

    logger = logging.getLogger("evaluation_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # IMPORTANT: avoid duplicate logs

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger