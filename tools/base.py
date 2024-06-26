from abc import ABC, abstractmethod

class BaseTool(ABC):

    @abstractmethod
    def invoke(self, *args, **kwargs):
        pass

