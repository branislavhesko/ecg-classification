from sklearn.metrics import confusion_matrix

class Metrics:

    def __init__(self) -> None:
        self.predictions = []
        self.labels = []

    def reset(self):
        self.predictions = []
        self.labels = []

    def update(self, prediction, label):
        prediction = prediction.tolist()
        label = label.tolist()
        self.predictions += prediction
        self.labels += label

    @property
    def sensitivity(self):
        pass

    @property
    def specificity(self):
        pass

    @property
    def recall(self):
        pass

    @property
    def accuracy(self):
        pass

    def confusion_matrix(self):
        return confusion_matrix(self.labels, self.predictions)
