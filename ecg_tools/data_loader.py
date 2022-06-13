from typing import Callable

import pandas as pd
import torch
import torch.utils.data as data

from ecg_tools.config import DatasetConfig, Mode


class EcgLoader(data.Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        label = int(self.annotations[item, -1])
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)
        return signal.unsqueeze(0), torch.tensor(label).long()


def get_data_loaders(config: DatasetConfig):
    return {
        Mode.train: data.DataLoader(
            EcgLoader(config.path[Mode.train], config.transforms[Mode.train]),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True
        ),
        Mode.eval: data.DataLoader(
            EcgLoader(config.path[Mode.eval], config.transforms[Mode.eval]),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False
        )
    }


if __name__ == "__main__":
    loader = get_data_loaders(DatasetConfig())
    for signal, label in loader[Mode.train]:
        from matplotlib import pyplot as plt
        plt.style.use("ggplot")
        plt.plot(signal.numpy()[0, :], "-r")
        plt.show()
        break
