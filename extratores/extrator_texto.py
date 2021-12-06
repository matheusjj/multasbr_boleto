import cv2
import pytesseract
from pytesseract import Output
from entidades.palavra import Palavra
from doctr.models import ocr_predictor


class TorchOCR:
    def __init__(self):
        None

    def __call__(self, img):
        model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

        result = model([img])

        palavras = []
        for bloco in result.export()['pages'][0]['blocks']:
            for num_linha, linha in enumerate(bloco['lines']):
                for palavra in linha['words']:
                    palavras.append(Palavra(palavra['value'], palavra['geometry'], num_linha, linha['geometry']))

        return palavras


class PytesseractOCR:
    def __init__(self, psm='4'):
        self.psm = psm

    def __call__(self, img):
        imagem = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        options = '-l {} --psm {}'.format('por', self.psm)
        palavras_capturadas = pytesseract.image_to_data(imagem,
                                                        output_type=Output.DICT, config=options)

        palavras = list()

        for idx in range(0, len(palavras_capturadas['text'])):
            if len(palavras_capturadas['text'][idx].strip()) != 0:
                palavras.append(Palavra(
                    palavras_capturadas['text'][idx],
                    (
                        (palavras_capturadas['top'][idx], palavras_capturadas['left'][idx]),

                        (palavras_capturadas['top'][idx] + palavras_capturadas['height'][idx],
                         palavras_capturadas['left'][idx] + palavras_capturadas['width'][idx])
                    ),
                    palavras_capturadas['line_num'][idx]
                ))

        return palavras
