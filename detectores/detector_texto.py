import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours

'''
Classes responsáveis para a determinação da região de texto de um documento.
Retorna um tuple com tuples dentro com as coordenadas do ponto do topo esquerdo e baixo direito.
'''


class DetectorProjecao:
    def __init__(self, proporcao_intervalo=0.1):
        self.proporcao_intervalo = proporcao_intervalo
        self.proporcao_texto = proporcao_intervalo - 0.04

    def __call__(self, imagem):
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        thresh[thresh == 0] = 0
        thresh[thresh == 255] = 1

        projecao_vertical = np.sum(thresh, axis=0)
        projecao_horizontal = np.sum(thresh, axis=1)

        regiao_vertical = self.__intervalo_deslizante(projecao_vertical,
                                                      self.proporcao_intervalo, self.proporcao_texto)
        regiao_horizontal = self.__intervalo_deslizante(projecao_horizontal,
                                                        self.proporcao_intervalo, self.proporcao_texto)

        topo_esquerdo = (regiao_vertical[0], regiao_horizontal[0])
        baixo_direito = (regiao_vertical[1], regiao_horizontal[1])

        return topo_esquerdo, baixo_direito

    @staticmethod
    def __intervalo_deslizante(projecao, proporcao_intervalo, proporcao_texto):
        limites_regiao = []
        intervalo = int(projecao.shape[0] * proporcao_intervalo)
        intervalo_texto = int(projecao.shape[0] * proporcao_texto)

        projecao_diff = np.array(np.diff(projecao), dtype='int16')
        projecao_diff[projecao_diff != 0] = 1

        for idx in range(projecao.shape[0]):
            if idx + intervalo >= projecao.shape[0]:
                break
            elif np.sum(projecao_diff[idx:idx+intervalo], axis=0) >= intervalo_texto:
                limites_regiao.append(idx)
                break

        projecao_diff = projecao_diff[::-1]
        for idx in range(projecao.shape[0]):
            if idx + intervalo >= projecao.shape[0]:
                break
            elif np.sum(projecao_diff[idx:idx+intervalo], axis=0) >= intervalo_texto:
                limites_regiao.append(projecao_diff.shape[0] - idx)
                break

        return limites_regiao[0], limites_regiao[1]


def pre_processamento_deteccao(imagem, tam_kernel):
    retangulo_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 * tam_kernel, tam_kernel))

    blurred = cv2.GaussianBlur(imagem, (3, 3), 0)
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, retangulo_kernel)

    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    grad = np.absolute(grad)
    (min_val, max_val) = (np.min(grad), np.max(grad))
    grad = (grad - min_val) / (max_val - min_val)
    grad = (grad * 255).astype('uint8')

    closed = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, retangulo_kernel)
    return cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


class DetectorPrimeiroPixelTexto:
    def __init__(self, tam_kernel=7):
        self.tam_kernel = tam_kernel

    def __call__(self, imagem):
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        quadrado_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 * self.tam_kernel, 3 * self.tam_kernel))

        thresh = pre_processamento_deteccao(cinza, tam_kernel=self.tam_kernel)
        area = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, quadrado_kernel)

        soma_x = np.sum(area, axis=0)
        soma_y = np.sum(area, axis=1)

        te_x, bd_x = self.__determinar_area_texto(soma_x)
        te_y, bd_y = self.__determinar_area_texto(soma_y)

        topo_esquerdo = (te_x, te_y)
        baixo_direito = (bd_x, bd_y)

        return topo_esquerdo, baixo_direito

    @staticmethod
    def __determinar_area_texto(linha):
        for (idx, val) in enumerate(linha):
            if val != 0:
                menor = idx
                break

        for (idx, val) in enumerate(reversed(linha)):
            if val != 0:
                maior = linha.shape[0] - idx - 1
                break

        return menor, maior


class DetectorContorno:
    def __init__(self, tam_kernel=7, tolerancia=0):
        self.tam_kernel = tam_kernel
        self.tolerancia = tolerancia

    def __call__(self, imagem):
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        (altura, largura) = cinza.shape

        quadrado_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3 * self.tam_kernel, 3 * self.tam_kernel))

        thresh = pre_processamento_deteccao(cinza, tam_kernel=self.tam_kernel)
        thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, quadrado_kernel, iterations=3)
        thresh_eroded = cv2.morphologyEx(thresh_closed, cv2.MORPH_ERODE, quadrado_kernel)

        cnts = cv2.findContours(thresh_eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method='top-to-bottom')[0]

        area_interesse = None

        for c in cnts:
            area_interesse = cv2.boundingRect(c)

        if area_interesse is not None:
            (x, y, w, h) = area_interesse

            x_esquerda = int((x - w * self.tolerancia))
            y_cima = int((y - h * self.tolerancia))
            x_direita = int((x + w + w * self.tolerancia))
            y_baixo = int((y + h + h * self.tolerancia))

            ponto_na_imagem = lambda local, tam: local if local < tam else tam

            topo_esquerdo = (ponto_na_imagem(x_esquerda, 0), ponto_na_imagem(y_cima, 0))
            baixo_direito = (ponto_na_imagem(x_direita, largura), ponto_na_imagem(y_baixo, altura))
        else:
            topo_esquerdo = (0, 0)
            baixo_direito = (altura, largura)

        return topo_esquerdo, baixo_direito
