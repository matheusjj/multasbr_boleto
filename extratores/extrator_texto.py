from entidades.palavra import Palavra
from doctr.models import ocr_predictor


class TorchOCR:
    def __init__(self):
        None

    def __call__(self, img):
        model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

        result = model([img])
        # result.show(doc)

        palavras = []
        for bloco in result.export()['pages'][0]['blocks']:
            for num_linha, linha in enumerate(bloco['lines']):
                for palavra in linha['words']:
                    palavras.append(Palavra(palavra['value'], palavra['geometry'], num_linha, linha['geometry']))

        return palavras
