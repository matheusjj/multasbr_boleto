# import argparse
# from extrator_info import ExtratorInfo
#
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--imagem', type=str, default='imagens/folha_1.jpeg',
#                 help='caminho da imagem')
# ap.add_argument('-m', '--modo', type=str, default='auto',
#                 help='modo detecção das fronteiras')
# args = vars(ap.parse_args())

import cv2
from pre_processadores import PreProcessador
from extratores.extrator_documento import ExtratorDocumento
from detectores.detector_documento import DetectorLRDECustomizado
from extratores.extrator_texto import TorchOCR
from detectores.detector_layout import DetectorLayout


class DevolveTexto:
    def __init__(self):
        self.pre_processamento = PreProcessador()  # Operações morfológicas de pre-processamento da imagem
        self.detector_documento = DetectorLRDECustomizado()  # Instância do Detector de documento por Hough
        self.extrator_documento = ExtratorDocumento()  # Instância do extrator do documento da imagem
        # self.detector_texto = DetectorPrimeiroPixelTexto()  # Instância do detector de texto no documento
        self.detector_layout = DetectorLayout()
        self.extrator_texto = TorchOCR()  # Instância do extrator de texto no documento

    def __call__(self, caminho):
        img_original = cv2.imread(caminho)

        img, ratio = self.pre_processamento(img_original)
        vertices = self.detector_documento(img.copy())
        documento = self.extrator_documento(img, vertices, img_original, ratio)
        # self.detector_layout(documento)
        palavras = self.extrator_texto(documento)
        # limite_texto = self.detector_texto(documento)
        # limite_texto = self.detector_texto(img_original)

        # return img_original[limite_texto[0][1]:limite_texto[1][1], limite_texto[0][0]:limite_texto[1][0]]
        # return documento[limite_texto[0][1]:limite_texto[1][1], limite_texto[0][0]:limite_texto[1][0]]
        return palavras


devolve = DevolveTexto()
# palavras = devolve('imagens/camera/103.jpeg')
palavras = devolve('imagens/pdf_4.png')
print(palavras)
