import cv2
import numpy as np
from entidades.roi_ajustavel import RoiAjustavel, arrastar_quad


# Classe para retornar a parte da imagem original selecionada pelo ROI (automatica ou manualmente).
class ExtratorDocumento:
    roi_selecionada = None

    def __init__(self, nome_janela='Selecao') -> None:
        self.nome_janela = nome_janela

    def __call__(self, imagem, quadrilatero, imagem_original, ratio, e_teste=False):
        if e_teste:
            return self.__transformacao_quatro_vertices(imagem_original, quadrilatero, ratio, e_teste)

        cv2.imshow(self.nome_janela, imagem)
        roi = RoiAjustavel(imagem, self.nome_janela, imagem.shape[1], imagem.shape[0], quadrilatero)
        cv2.setMouseCallback(self.nome_janela, arrastar_quad, roi)
        cv2.namedWindow(self.nome_janela)

        quadrilatero_roi = roi.selecionar_roi(imagem)

        return self.__transformacao_quatro_vertices(imagem_original, quadrilatero_roi, ratio)

    @staticmethod
    def __transformacao_quatro_vertices(imagem, roi, ratio, e_teste=False):
        pontos_ajustados = roi.retornar_vertices() * ratio

        if e_teste:
            return pontos_ajustados

        (tl, tr, br, bl) = pontos_ajustados

        # Determina as larguras do retângulo e determina o maior
        largura_baixo = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        largura_cima = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        largura_max = max(int(largura_baixo), int(largura_cima))

        # Determina as alturas do retângulo e determina o maior
        altura_direita = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        altura_esquerda = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        altura_max = max(int(altura_direita), int(altura_esquerda))

        # Nova matriz para a qual o documento será transformado
        matriz_documento = np.array([
            [0, 0],
            [largura_max - 1, 0],
            [largura_max - 1, altura_max - 1],
            [0, altura_max - 1]
        ], dtype='float32')

        matriz_transf = cv2.getPerspectiveTransform(pontos_ajustados, matriz_documento)
        deslocado = cv2.warpPerspective(imagem, matriz_transf, (largura_max, altura_max))

        return deslocado
