import torch

class Flatten():
    def __call__(self, image):
        return image.view(-1)


class StaticBinarize():
    def __call__(self, image):
        return image.round().long()


class DynamicBinarize():
    def __call__(self, image):
        return image.bernoulli().long()
