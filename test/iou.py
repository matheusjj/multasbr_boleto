import cv2
import glob
from pre_processadores import PreProcessador
from extratores.extrator_documento import IoUExtratorDocumento
from detectores.detector_documento import DetectorDocumentoHough
from detectores.detector_texto import DetectorPrimeiroPixelTexto


class DevolveTexto:
    def __init__(self):
        self.pre_processamento = PreProcessador()  # Operações morfológicas de pre-processamento da imagem
        self.detector_documento = DetectorDocumentoHough()  # Instância do Detector de documento por Hough
        self.extrator_documento = IoUExtratorDocumento()  # Instância do extrator do documento da imagem
        self.detector_texto = DetectorPrimeiroPixelTexto()  # Instância do detector de texto no documento

    def __call__(self, caminho):
        img_original = cv2.imread(caminho)

        img, ratio = self.pre_processamento(img_original)
        vertices = self.detector_documento(img.copy())
        documento = self.extrator_documento(img, vertices, img_original, ratio)

        return documento


caminho_input = '../imagens/camera/'
caminho_output = '../imagens/camera/mascara_doc_detectado/'

arquivos = glob.glob(caminho_input + '*.jpg')
devolve = DevolveTexto()

for arquivo in arquivos:
    print(arquivo)
    imagem_processada = devolve(arquivo)
    cv2.imwrite(caminho_output + arquivo.split('/')[-1], imagem_processada)

# for arq in arquivos:
#     print(arq.split('/')[-1])
