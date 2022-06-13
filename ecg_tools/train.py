import einops
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from tqdm import tqdm

from ecg_tools.config import EcgConfig, Mode
from ecg_tools.data_loader import get_data_loaders
from ecg_tools.metrics import Metrics
from ecg_tools.model import ECGformer


class ECGClassifierTrainer:

    def __init__(self, config: EcgConfig) -> None:
        self.model = ECGformer(
            embed_size=config.model.embed_size,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            num_classes=config.model.num_classes,
            signal_length=config.model.signal_length,
            expansion=config.model.expansion,
            input_channels=config.model.input_channels
        ).to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.4, 0.2, 0.5, 0.2]).to(self.config.device))
        self.data_loader = get_data_loaders(self.config.dataset)
        self.metrics = {
            Mode.train: Metrics(),
            Mode.eval: Metrics()
        }

    def train(self):
        confusion_matrix_image_train, confusion_matrix_image_eval = np.zeros((1, 1, 3)), np.zeros((1, 1, 3))
        for epoch in range(self.config.num_epochs):
            confusion_matrix_image_train = self.train_epoch(epoch)

            if epoch % self.config.validation_frequency == 0:
                confusion_matrix_image_eval = self.validate_epoch(epoch)
        return {
            Mode.train: confusion_matrix_image_train,
            Mode.eval: confusion_matrix_image_eval
        }

    def train_epoch(self, epoch):
        self.model.train()
        loader = tqdm(self.data_loader[Mode.train])
        accuracy = 0
        self.metrics[Mode.train].reset()
        for index, data in enumerate(loader):
            self.optimizer.zero_grad()
            signal, label = [d.to(self.config.device) for d in data]
            prediction = self.model(einops.rearrange(signal, "b c e -> b e c"))
            loss = self.loss(prediction, label)
            loss.backward()
            self.optimizer.step()
            accuracy += torch.sum(prediction.argmax(1) == label)
            self.metrics[Mode.train].update(prediction.argmax(1), label)
            loader.set_description(f"TRAINING: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}")
        print(f"TRAINING Accuracy: {accuracy / len(loader) / self.config.dataset.batch_size}")
        print(self.metrics[Mode.train].confusion_matrix())
        return self.metrics[Mode.train].confusion_matrix_image()

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        accuracy = 0
        loader = tqdm(self.data_loader[Mode.eval])
        self.metrics[Mode.eval].reset()
        for index, data in enumerate(loader):
            signal, label = [d.to(self.config.device) for d in data]
            prediction = self.model(einops.rearrange(signal, "b c e -> b e c"))
            loss = self.loss(prediction, label)
            accuracy += torch.sum(prediction.argmax(1) == label)
            self.metrics[Mode.eval].update(prediction.argmax(1), label)
            loader.set_description(f"VALIDATION: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}")
        print(f"VALIDATION Accuracy: {accuracy / len(loader) / self.config.dataset.batch_size}")
        print(self.metrics[Mode.eval].confusion_matrix())
        return self.metrics[Mode.eval].confusion_matrix_image()


if __name__ == "__main__":
    cm = ECGClassifierTrainer(EcgConfig()).train()
    plt.figure(dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(cm[Mode.train])
    plt.subplot(1, 2, 2)
    plt.imshow(cm[Mode.eval])
    plt.savefig("training_results.png")