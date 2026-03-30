from abc import ABC, abstractmethod


class Plot(ABC):
    @abstractmethod
    def draw(self):
        raise NotImplementedError()

    @abstractmethod
    def dump(self, *args, **kwargs):
        raise NotImplementedError()
