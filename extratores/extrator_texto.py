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

        resultado = model([img])

        palavras = []
        resultado = resultado.export()
        page_alt, page_lar = resultado['pages'][0]['dimensions']
        for num_bloco, bloco in enumerate(resultado['pages'][0]['blocks']):
            for num_linha, linha in enumerate(bloco['lines']):
                for palavra in linha['words']:
                    palavras.append(Palavra(
                        palavra['value'],
                        ((int(palavra['geometry'][0][0] * page_lar), int(palavra['geometry'][0][1] * page_alt)),
                         (int(palavra['geometry'][1][0] * page_lar), int(palavra['geometry'][1][1] * page_alt))),
                        num_linha,
                        ((int(linha['geometry'][0][0] * page_lar), int(linha['geometry'][0][1] * page_alt)),
                         (int(linha['geometry'][1][0] * page_lar), int(linha['geometry'][1][1] * page_alt))),
                        num_bloco
                    ))

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
