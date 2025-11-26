from abc import ABC, abstractmethod


class TrainingTorchModel(ABC):
    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
