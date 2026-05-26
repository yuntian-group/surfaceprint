from typing import List
class BaseDetector:
    def batch_predict(self, x: List[str]):
        raise NotImplemented()

    def predict(self, x: str):
        raise NotImplemented()