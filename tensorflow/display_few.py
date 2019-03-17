from utils import *

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name")
args = parser.parse_args()

IMG_PATH = '/data/home/oliver/git/generative_autoencoders/images'

name=args.name

display(os.path.join(IMG_PATH, name))