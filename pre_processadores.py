import cv2.cv2 as cv2
import imutils


class PreProcessador:
    def __init__(self, altura=800):
        self.altura = altura

    def __call__(self, imagem):
        img = imutils.resize(imagem, height=self.altura, inter=cv2.INTER_CUBIC)
        ratio = imagem.shape[0] / float(self.altura)

        return img, ratio
