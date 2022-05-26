import einops
import torch
from tqdm import tqdm

from ecg_tools.config import EcgConfig, Mode
from ecg_tools.data_loader import get_data_loaders
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
        self.loss = torch.nn.CrossEntropyLoss()
        self.data_loader = get_data_loaders(self.config.dataset)

    def train(self):

        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        self.model.train()
        loader = tqdm(self.data_loader[Mode.train])
        accuracy = 0
        for index, data in enumerate(loader):
            self.optimizer.zero_grad()
            signal, label = [d.to(self.config.device) for d in data]
            prediction = self.model(einops.rearrange(signal, "b c e -> b e c"))
            loss = self.loss(prediction, label)
            loss.backward()
            self.optimizer.step()
            accuracy += torch.sum(prediction.argmax(1) == label)
            loader.set_description(f"Epoch: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}")
        print(f"Final Accuracy: {accuracy / len(loader) / self.config.dataset.batch_size}")

if __name__ == "__main__":
    ECGClassifierTrainer(EcgConfig()).train()
