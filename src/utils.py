import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--mats_dir", type=str, required=True)
    parser.add_argument("--transform", action="store_true")

    return parser.parse_args()
