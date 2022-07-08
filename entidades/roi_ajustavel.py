import cv2 as cv2
import numpy as np
from entidades.quadrilatero import Quadrilatero

'''
Classe para a criação e manipulação do ROI pelo usuário
Retorna um RoiAjustavel utilizado na extração do documento da imagem selecionada
'''


class RoiAjustavel:
    # Dimensões do canvas
    dimensoes_canvas = Quadrilatero()

    # Dimensões do ROI
    dimensoes_roi = Quadrilatero()

    # Imagem
    imagem = None

    # Nome da janela
    nome_janela = ''

    # Valores
    poligono = None
    raio = 7
    ponto_id = None
    primeiro_ponto = [None, None]

    # Flags
    return_flag = False
    ativo_flag = False
    arrastar_flag = False
    segurar_flag = False

    def __init__(self, imagem, nome_janela, lar_janela, alt_janela, quadrilatero) -> None:
        self.imagem = imagem
        self.nome_janela = nome_janela
        self.poligono = np.zeros(imagem.shape[:2], dtype='uint8')

        self.dimensoes_canvas.te = [0, 0]
        self.dimensoes_canvas.td = [lar_janela, 0]
        self.dimensoes_canvas.bd = [lar_janela, alt_janela]
        self.dimensoes_canvas.be = [0, alt_janela]

        if not quadrilatero.e_zerado():
            vertices = quadrilatero.retornar_vertices()
            self.dimensoes_roi.te = vertices[0]
            self.dimensoes_roi.td = vertices[1]
            self.dimensoes_roi.bd = vertices[2]
            self.dimensoes_roi.be = vertices[3]

            if self.dimensoes_roi.e_quad():
                self.ativo_flag = True
        else:
            self.dimensoes_roi.te = [0, 0]
            self.dimensoes_roi.td = [lar_janela, 0]
            self.dimensoes_roi.bd = [lar_janela, alt_janela]
            self.dimensoes_roi.be = [0, alt_janela]

    def selecionar_roi(self, imagem):
        key = 0

        while not key == 32:
            cor = (0, 255, 0)
            temp = imagem.copy()
            verts = np.array(self.dimensoes_roi.retornar_vertices(), dtype='int32')
            cv2.polylines(temp, [verts], True, cor, 2)

            (te, td, bd, be) = verts

            for vertice in (te, td, bd, be):
                cv2.circle(temp, (vertice[0], vertice[1]), self.raio, cor, -1)

            dx_te, dx_td, dx_bd, dx_be = self.dimensoes_roi.x_diff()
            dy_te, dy_td, dy_bd, dy_be = self.dimensoes_roi.y_diff()

            cv2.circle(temp, (te[0] + int(dx_te / 2), te[1] + int(dy_te / 2)), 10, cor, -1)
            cv2.circle(temp, (bd[0] + int(dx_bd / 2), bd[1] + int(dy_bd / 2)), 10, cor, -1)
            cv2.circle(temp, (be[0] + int(dx_be / 2), be[1] + int(dy_be / 2)), 10, cor, -1)
            cv2.circle(temp, (td[0] + int(dx_td / 2), td[1] + int(dy_td / 2)), 10, cor, -1)

            # cv2.line(temp,
            #          (te[0] + int(dx_te / 2), te[1] + int(dy_te / 2)),
            #          (bd[0] + int(dx_bd / 2), bd[1] + int(dy_bd / 2)),
            #          cor, 2)
            # cv2.line(temp,
            #          (be[0] + int(dx_be / 2), be[1] + int(dy_be / 2)),
            #          (td[0] + int(dx_td / 2), td[1] + int(dy_td / 2)),
            #          cor, 2)

            cv2.imshow(self.nome_janela, temp)

            key = cv2.waitKey(30)
            if key == 'c' or key == 'C' or self.return_flag:
                break

        cv2.destroyAllWindows()

        return self.dimensoes_roi


# Funções complementares
def arrastar_quad(event, x, y, flags, roi):
    if event == cv2.EVENT_LBUTTONDOWN:
        roi.primeiro_ponto = [x, y]

        roi.ponto_id = determinar_marcador(x, y, roi)
        if roi.ponto_id == -1:
            if not esta_proximo_quadrilatero(x, y, roi):
                temporario = np.zeros(roi.imagem.shape[:2], dtype='uint8')
                roi.poligono[roi.poligono == 255] = 0
                verts = np.array(roi.dimensoes_roi.retornar_vertices(), dtype='int32')
                cv2.fillPoly(roi.poligono, [verts], 255)

                if temporario[x, y] == 0:
                    roi.ativo_flag = True
                    roi.arrastar_flag = True

        else:
            roi.segurar_flag = True

    elif event == cv2.EVENT_LBUTTONUP:
        e_quad = roi.dimensoes_roi.e_quad()
        roi.arrastar_flag = not e_quad
        roi.ativo_flag = e_quad

        roi.segurar_flag = False
        roi.ponto_id = None

    elif roi.ativo_flag and event == cv2.EVENT_MOUSEMOVE:
        mouse_moveu(x, y, roi)

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        roi.return_flag = True


def mouse_moveu(mouse_x, mouse_y, quad_roi):
    if mouse_x > quad_roi.dimensoes_canvas.bd[0]:
        mouse_x = quad_roi.dimensoes_canvas.bd[0] - 5
    elif mouse_x < quad_roi.dimensoes_canvas.te[0]:
        mouse_x = quad_roi.dimensoes_canvas.te[0] + 5

    if mouse_y > quad_roi.dimensoes_canvas.bd[1]:
        mouse_y = quad_roi.dimensoes_canvas.bd[1] - 5
    elif mouse_y < quad_roi.dimensoes_canvas.te[1]:
        mouse_y = quad_roi.dimensoes_canvas.te[1] + 5

    if quad_roi.arrastar_flag:
        quad_roi.dimensoes_roi.te = quad_roi.primeiro_ponto
        quad_roi.dimensoes_roi.td = (mouse_x, quad_roi.dimensoes_roi.te[1])
        quad_roi.dimensoes_roi.bd = (mouse_x, mouse_y)
        quad_roi.dimensoes_roi.be = (quad_roi.dimensoes_roi.te[0], mouse_y)

    elif quad_roi.segurar_flag:
        if quad_roi.dimensoes_roi.e_concavo(quad_roi.ponto_id):
            quad_roi.segurar_flag = False

        if quad_roi.ponto_id == 0:
            quad_roi.dimensoes_roi.te = (mouse_x, mouse_y)
        elif quad_roi.ponto_id == 1:
            quad_roi.dimensoes_roi.td = (mouse_x, mouse_y)
        elif quad_roi.ponto_id == 2:
            quad_roi.dimensoes_roi.bd = (mouse_x, mouse_y)
        elif quad_roi.ponto_id == 3:
            quad_roi.dimensoes_roi.be = (mouse_x, mouse_y)


def determinar_marcador(mouse_x, mouse_y, roi):
    for idx, vertice in enumerate(roi.dimensoes_roi.retornar_vertices()):
        dist = np.sqrt(((vertice[0] - mouse_x) ** 2) + ((vertice[1] - mouse_y) ** 2))

        if dist <= roi.raio:
            return idx

    return -1


def esta_proximo_quadrilatero(mouse_x, mouse_y, roi):
    distancias = []
    for idx, vertice in enumerate(roi.dimensoes_roi.retornar_vertices()):
        distancias.append(np.sqrt(((vertice[0] - mouse_x) ** 2) + ((vertice[1] - mouse_y) ** 2)))

    distancias.sort()
    return distancias[0] <= roi.imagem.shape[1] * 0
    # return distancias[0] <= roi.imagem.shape[1] * 0.1
