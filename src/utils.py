import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--mats_dir", type=str, required=True)
    parser.add_argument("--transform", action="store_true")

    parser.add_argument("--checkpoint", type=str, help="path to saved model")

    return parser.parse_args()
