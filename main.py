# import argparse
# from extrator_info import ExtratorInfo
#
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--imagem', type=str, default='imagens/folha_1.jpeg',
#                 help='caminho da imagem')
# ap.add_argument('-m', '--modo', type=str, default='auto',
#                 help='modo detecção das fronteiras')
# args = vars(ap.parse_args())

import cv2.cv2 as cv2
import pytesseract
from pre_processadores import PreProcessador
from extratores.extrator_documento import ExtratorDocumento
from detectores.detector_documento import DetectorDocumentoHough
from detectores.detector_texto import DetectorPrimeiroPixelTexto


class DevolveTexto:
    def __init__(self):
        self.pre_processamento = PreProcessador()  # Operações morfológicas de pre-processamento da imagem
        self.detector_documento = DetectorDocumentoHough()  # Instância do Detector de documento por Hough
        self.extrator_documento = ExtratorDocumento()  # Instância do extrator do documento da imagem
        self.detector_texto = DetectorPrimeiroPixelTexto()  # Instância do detector de texto no documento

    def __call__(self, caminho):
        img_original = cv2.imread(caminho)

        img, ratio = self.pre_processamento(img_original)
        vertices = self.detector_documento(img.copy())
        documento = self.extrator_documento(img, vertices, img_original, ratio)
        limite_texto = self.detector_texto(documento)

        return documento[limite_texto[0][1]:limite_texto[1][1], limite_texto[0][0]:limite_texto[1][0]]


t = DevolveTexto()
t = t('imagens/pdf_2.png')
t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
options = '-l {} --psm {}'.format('por', '4')
text = pytesseract.image_to_string(t, config=options)
print(text)
