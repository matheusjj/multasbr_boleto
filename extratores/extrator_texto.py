import numpy as np


class ExtratorTexto:
    def __init__(self):
        pass

    def __call__(self, imagem, vertices, ratio):
        vertices = np.array(vertices * ratio, dtype='uint32').reshape(2, 2)

        return imagem[vertices[0][1]:vertices[1][1], vertices[0][0]:vertices[1][0]]
