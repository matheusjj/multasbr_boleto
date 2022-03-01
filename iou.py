import os
import cv2
import csv
import numpy as np

from pre_processadores.pre_processador import PreProcessador
from detectores.detector_documento import DetectorLRDECustomizado
from extratores.extrator_documento import ExtratorDocumento


def detector_lrde():
    caminho_img = '/home/matheus/Documentos/Exemplo/'
    caminho_mascara_lrde = '/home/matheus/Documentos/Exemplo/Mascara_LRDE/'

    pre_processador = PreProcessador()
    detector = DetectorLRDECustomizado()
    extrator = ExtratorDocumento()

    for arquivo in os.listdir(caminho_img):
        # if os.path.isfile(os.path.join(caminho_img, arquivo)) and arquivo == 'D43_F3.jpg':
        if os.path.isfile(os.path.join(caminho_img, arquivo)):
            print(arquivo)
            original = cv2.imread(caminho_img + arquivo)

            img, ratio = pre_processador(original)
            vertices = detector(img.copy())
            documento = extrator(img, vertices, original, ratio)
            quad = np.array([documento], dtype='int32')

            img_final = np.zeros(original.shape[:2])

            cv2.fillPoly(img_final, [quad], 255)
            cv2.imwrite(caminho_mascara_lrde + arquivo, img_final)


def processar_iou():
    caminho_auto = '/home/matheus/Documentos/Exemplo/Mascara_LRDE/'
    caminho_manual = '/home/matheus/Documentos/Exemplo/Mascara/'
    arquivo_output = 'Resultado_IoU_Hough.csv'
    resultados = []

    for arquivo in os.listdir(caminho_auto):
        if os.path.isfile(os.path.join(caminho_auto, arquivo)):
            auto = cv2.imread(caminho_auto + arquivo)
            manual = cv2.imread(caminho_manual + arquivo)

            intersecao = cv2.bitwise_and(auto, manual)
            uniao = cv2.bitwise_or(auto, manual)

            iou = np.count_nonzero(intersecao == 255) / np.count_nonzero(uniao == 255)
            resultados.append([arquivo, iou])

    # print(resultados)
    campos = ['Imagem', 'IoU']

    with open(arquivo_output, 'w') as f:
        write = csv.writer(f)
        write.writerow(campos)
        write.writerows(resultados)


detector_lrde()
# processar_iou()
