"""
    OVAD: Speech detector in adversarial conditions

    Copyright 2023 by Lukasz Smietanka and Tomasz Maka

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
"""

import sys
import librosa
import argparse
from pathlib import Path

from ovad import Ovad


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("filename", type=str, help="input audio file")

optional = parser.add_argument_group('optional arguments')
optional.add_argument("-h", "--help", action="help",
                      help="show this help message and exit")
optional.add_argument("-p", "--to-pdf", action="store_true",
                      help="saves PDF figure with marked speech segments")
optional.add_argument("-c", "--to-csv", action="store_true",
                      help="stores speech segments to CSV file")

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(-1)

models_dir = (Path(__file__).resolve()).parent.joinpath("models")

args = parser.parse_args()
in_filename = Path(args.filename)

if not in_filename.exists():
    print("ERROR: File not found")
    sys.exit(-1)

samples, fs = librosa.load(in_filename, sr=None)

ovad = Ovad("AugViT", models_dir=models_dir)

predict = ovad.predict(filename=in_filename)

show = True

if args.to_csv is True:
    ovad.segments_to_csv(out_file=in_filename.stem + "-segments.csv")
    show = False

if args.to_pdf is True:
    ovad.plot(save=True, out_file=in_filename.stem + "-segments.pdf")
    show = False

if show:
    ovad.plot()
