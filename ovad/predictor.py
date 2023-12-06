"""
     OVAD: Speech detector in adversarial conditions

     Copyright 2023 by Lukasz Smietanka and Tomasz Maka

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.
"""

import librosa
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import importlib.util

models_list = ["AugViT"]
models_repo = Path("models")


class Ovad:
    def __init__(self, model_type: str, device: str = 'cpu',
                 models_dir: Path = models_repo):
        """ Initializes the instance of the predictor.

        Args:
          model_type: model name
          device: device type used to allocate the torch data
        """

        if model_type not in models_list:
            raise ValueError("ERROR: Incorrect model name")

        model_path = models_dir.joinpath(f"{model_type}.pth")

        self.model_info = torch.load(model_path,
                                     map_location=torch.device(device))

        spec = importlib.util.spec_from_file_location(
            model_type, f"{models_dir}/{model_type}.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)

        self.model = model.Model(*self.model_info['data']['shape'],
                                 self.model_info['sr'],
                                 **self.model_info['model']['args'],
                                 device=device)
        self.model.load_state_dict(self.model_info['model']['state'])

        self.model = self.model.to(device)

        self.device = device
        self.fs = self.model_info['sr']

        self.samples = None
        self.predicted_data = None
        self.voice_segments = None

    def predict(self, samples: np.ndarray = None, sr: int = None,
                filename: str = None) -> list:
        """Generate predicted vector of presence speech in given audio data.

        Args:
          samples: audio data
          sr: sampling rate [Hz]
          filename: audio file

        Returns:
          binary vector of speech presence in every one-second window
          (1: speech, 0: non-speech)
        """

        if filename:
            self.samples, sr = librosa.load(filename, sr=None)
        elif samples is not None and sr is not None:
            self.samples = samples
        else:
            raise ValueError("ERROR: Incorrect arguments")

        if sr != self.fs:
            self.samples = librosa.resample(samples, sr, self.fs)

        if len(self.samples) / self.fs < 1.0:
            raise ValueError("ERROR: file is too short (len < 1s)")

        len_in_sec = int(np.floor(len(self.samples) / self.fs))
        frames = np.split(self.samples[:int(self.fs * len_in_sec)], len_in_sec)
        self.predicted_data = []

        for sec in range(len_in_sec):
            x = torch.from_numpy(frames[sec])[None, :]
            self.predicted_data.append(self.__predict__(x))

        self.get_speech_segments()

        return self.predicted_data

    def __predict__(self, x: torch.Tensor) -> int:
        """ Private method to predict speech presence in one-second part of
            audio data.

        Args:
          x: tensor with one-second audio data

        Returns:
          class of speech presence in input data: (1: speech, 0: non-speech)
        """

        self.model.eval()
        with torch.no_grad():
            sample = x.to(self.device)
            prediction = self.model(sample)["prob"]
        return int(torch.round(prediction).item())

    def get_speech_segments(self) -> list:
        """Generate a list of beginnings and ends detected speech segments
           based on the predicted vector of speech presence in each second of
           audio data.

        Args:

        Returns:
          list of pairs containing beginnings and ends of detected
          speech segments
        """

        active = False
        self.voice_segments = []
        tmp = []
        for i, v in enumerate(self.predicted_data):
            state = v
            if state == 1 and not active:
                active = True
                tmp.append(i)
            if (state == 0 and active) or \
               (i == len(self.predicted_data) - 1 and active):
                active = False
                tmp.append(i + 1)
                self.voice_segments.append([tmp[0], tmp[1] - 1])
                tmp = []

        return self.voice_segments

    def segments_to_csv(self, out_file: str = "voice_segments.csv") -> int:
        """ Save a list of beginnings and ends of detected speech segments to
            the csv file.

        Args:
          out_file: filename to which the voice segments are saved

        Returns:
          number of detected segments

        """

        if len(self.voice_segments) == 0:
            raise ValueError("ERROR: No segments detected.")

        pd.DataFrame(np.asarray(self.voice_segments), columns=['begin', 'end'],
                     index=None).to_csv(out_file, index=False)

        return len(self.voice_segments)

    def plot(self, save: bool = False, plot: bool = True,
             out_file: str = "voice_segments.pdf"):
        """Generate a plot that presents given audio data and detected
           speech segments.

        Args:
          save: allow saving plot to pdf file, if False, only display using
                matplotlib
          out_file: filename to which the plot is saved

        Returns:

        """

        if len(self.voice_segments) == 0:
            raise ValueError("ERROR: No segments detected.")

        fig, axs = plt.subplots(1, 1, figsize=(10, 5))

        axs.plot(np.linspace(0, len(self.samples) / self.fs,
                             len(self.samples)),
                 self.samples, color="#c1c1c1")

        axs.set_xlabel("Time [s]")

        for seg in self.voice_segments:
            cc = PatchCollection(
                [Rectangle((seg[0], -1.1), seg[1] - seg[0], 2.2,
                           edgecolor='blue', facecolor='#81cff1')],
                zorder=2, alpha=0.5)
            axs.add_collection(cc)

            axs.axvline(x=seg[1], color='b')
            axs.axvline(x=seg[0], color='b')

        if save:
            plt.savefig(out_file, format='pdf')
        else:
            plt.show()
