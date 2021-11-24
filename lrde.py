import os
import csv
import cv2
import imutils
import numpy as np
from entidades.quadrilatero import Quadrilatero
from detectores.detector_documento import DetectorLRDECustomizado

# altura_redim = 800
# caminho = 'imagens/camera/'

# for arquivo in os.listdir(caminho):
#     if os.path.isfile(os.path.join(caminho, arquivo)):
#         # print(arquivo)
#
#         img = cv2.imread(caminho + arquivo)
#         alt, lar = img.shape[:2]
#         canvas = np.zeros([alt, lar], dtype='uint8')
#
#         img = imutils.resize(img, height=altura_redim)
#         ratio = alt / altura_redim
#
#         detector = DetectorLRDECustomizado()
#         quad = detector(img)
#
#         if quad is None:
#             quad = Quadrilatero()
#
#         pontos_ajustados = np.array(quad.retornar_vertices() * ratio, dtype='int32')
#         cv2.fillPoly(canvas, [pontos_ajustados], 255)
#
#         cv2.imwrite('imagens/camera/mascara_doc_detectado/' + arquivo, canvas)

caminho_auto = 'imagens/camera/mascara_doc_detectado/'
caminho_manual = 'imagens/camera/mascara_documento/'
resultados = []

for arquivo in os.listdir(caminho_auto):
    if os.path.isfile(os.path.join(caminho_auto, arquivo)):
        auto = cv2.imread(caminho_auto + arquivo)
        manual = cv2.imread(caminho_manual + arquivo)

        intersecao = cv2.bitwise_and(auto, manual)
        uniao = cv2.bitwise_or(auto, manual)

        iou = np.count_nonzero(intersecao == 255) / np.count_nonzero(uniao == 255)
        resultados.append([arquivo, iou])

print(resultados)
campos = ['Imagem', 'IoU']

with open('Resultado_IoU', 'w') as f:
    write = csv.writer(f)
    write.writerow(campos)
    write.writerows(resultados)
