import cv2
from doctr.models import ocr_predictor


class Palavra:
    def __init__(self, valor, localizacao, linha, linha_localizacao):
        self.valor = valor
        self.localizacao = localizacao
        self.linha = linha
        self.linha_localizacao = linha_localizacao


model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
img = cv2.imread('imagens/pdf_2.png')

result = model([img])
# result.show(doc)

palavras = []
for bloco in result.export()['pages'][0]['blocks']:
    print(bloco['geometry'])
    for num_linha, linha in enumerate(bloco['lines']):
        for palavra in linha['words']:
            palavras.append(Palavra(palavra['value'], palavra['geometry'], num_linha, linha['geometry']))

print(palavras)