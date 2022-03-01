import cv2
from pre_processadores.pre_processador import PreProcessador
from detectores.detector_documento import DetectorLRDECustomizado
from extratores.extrator_documento import ExtratorDocumento
from extratores.extrator_texto import TorchOCR, PytesseractOCR
from extratores.extrator_informacao import IERegrado


class RetirarDadosNotificacao:
    def __init__(self):
        self.pre_processamento = PreProcessador()  # Operações morfológicas de pre-processamento da imagem
        self.detector_documento = DetectorLRDECustomizado()  # Instância do Detector de documento por Hough
        self.extrator_documento = ExtratorDocumento()  # Instância do extrator do documento da imagem
        self.extrator_texto = PytesseractOCR()  # Instância do extrator de texto no documento
        self.extrator_informacao = IERegrado()  # Instância de extrator de informações relevantes

    def __call__(self, caminho):
        img_original = cv2.imread(caminho)

        img, ratio = self.pre_processamento(img_original)
        vertices = self.detector_documento(img.copy())
        documento = self.extrator_documento(img, vertices, img_original, ratio)
        palavras = self.extrator_texto(documento)
        palavras = self.extrator_texto(img_original)
        return palavras


devolve = RetirarDadosNotificacao()
devolve('imagens/Notificacoes/1.png')
