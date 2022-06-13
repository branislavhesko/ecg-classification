import torch


class Compose:

    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal


class RandomShift:

    def __init__(self, max_num_samples: int, probability=0.5) -> None:
        self._m = max_num_samples
        self._p = probability

    def __call__(self, signal):
        if torch.rand(1) > self._p:
            return signal

        return torch.roll(signal, torch.randint(-self._m, self._m, (1, )).item(), 0)


class RandomNoise:

    def __init__(self, max_amplitude, probability) -> None:
        self._a = max_amplitude
        self._p = probability

    def __call__(self, signal):
        if torch.rand(1) > self._p:
            return signal

        return signal + (torch.rand(len(signal)) - 0.5) * 2. * self._a
