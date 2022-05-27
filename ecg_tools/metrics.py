class Metrics:

    def __init__(self) -> None:
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0
        
    def reset(self):
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.tn = 0
        
    def update(prediction, label):
        pass

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
