from abc import ABC,abstractmethod
class SparseDiffusionBackend(ABC):
 @abstractmethod
 def generate(self,request):raise NotImplementedError
