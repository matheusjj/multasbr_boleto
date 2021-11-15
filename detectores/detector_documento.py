import cv2.cv2 as cv2
import imutils
import numpy as np
from math import atan
from sklearn.cluster import KMeans
from itertools import combinations
from entidades.quadrilatero import Quadrilatero

'''
Classes responsáveis para a determinação da região do documento na imagem.
Retorna uma instância da classe Quadrilatero ou None. 
'''


def pre_processamento_deteccao(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel)

    morph = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
    morph = cv2.normalize(morph, morph, 0, 1.0, cv2.NORM_MINMAX)
    morph = (morph * 255).astype('uint8')

    thresh_gauss = cv2.threshold(morph, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]

    blur = cv2.GaussianBlur(thresh_gauss, (5, 5), 0)
    return imutils.auto_canny(blur, sigma=0.5)


class DetectorDocumentoCanny:
    def __init__(self) -> None:
        self

    def __call__(self, imagem):
        canny = pre_processamento_deteccao(imagem)

        cnts = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            perimetro = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimetro, True)
            print(len(approx))

            if len(approx) == 4:
                contorno = approx
                break

        try:
            return Quadrilatero(contorno.reshape(4, 2))
        except UnboundLocalError:
            return None


class DetectorDocumentoHough:
    def __init__(self, angulo_agudo_lim=75.0, angulo_obtuso_lim=105.0, votos_hough=100,
                 ang_diagonal_inf=170.0, ang_diagonal_sup=190.0):
        self.angulo_agudo = angulo_agudo_lim
        self.angulo_obtuso = angulo_obtuso_lim
        self.votos_hough = votos_hough
        self.ang_diagonal_inf = ang_diagonal_inf
        self.ang_diagonal_sup = ang_diagonal_sup

    def __call__(self, imagem):
        canny = pre_processamento_deteccao(imagem)
        linhas = cv2.HoughLines(canny, 1, np.pi / 180, self.votos_hough)

        if linhas is None:
            return None

        intersecoes = []
        linhas_agrupadas = combinations(range(len(linhas)), 2)
        x_no_intervalo = lambda x: 0 <= x <= imagem.shape[1]
        y_no_intervalo = lambda y: 0 <= y <= imagem.shape[0]

        for i, j in linhas_agrupadas:
            linha_i, linha_j = linhas[i][0], linhas[j][0]

            if self.angulo_agudo < self.__determinar_angulo_entre_retas(linha_i, linha_j) < self.angulo_obtuso:
                int_point = self.__intersecao(linha_i, linha_j)

                if x_no_intervalo(int_point[0][0]) and y_no_intervalo(int_point[0][1]):
                    intersecoes.append(int_point)

        vertices = self.__encontrar_vertices(intersecoes)
        vertices = self.__determinar_vertices_proximos(vertices)

        if vertices is None:
            return None

        if len(vertices) <= 1:
            return None
        elif len(vertices) == 2:
            vertices = self.completar_diagonal(vertices)
            if vertices is None:
                return None
            else:
                return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))
        elif len(vertices) == 3:
            self.completar_vertices(vertices)
        else:
            return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))

    def completar_diagonal(self, vertices):
        angulo_entre_pontos = self.angulo_entre_pontos(vertices)

        if self.ang_diagonal_inf <= angulo_entre_pontos <= self.ang_diagonal_sup:
            vert_3 = [vertices[0][0], vertices[1][1]]
            vert_4 = [vertices[1][0], vertices[0][1]]
            return [vertices[0], vertices[1], vert_3, vert_4]
        else:
            return None

    def completar_vertices(self, vertices):
        vert_1, vert_2, vert_3 = vertices[0], vertices[1], vertices[2]

        ang_vert1_vert2 = self.angulo_entre_pontos([vert_1, vert_2])
        ang_vert1_vert3 = self.angulo_entre_pontos([vert_1, vert_3])

        if self.ang_diagonal_inf <= ang_vert1_vert2 <= self.ang_diagonal_sup:
            vert_4 = [-1 * vert_3[0][0], -1 * vert_3[0][1]]
            return [vert_1, vert_2, vert_3, vert_4]
        elif self.ang_diagonal_inf <= ang_vert1_vert3 <= self.ang_diagonal_sup:
            vert_4 = [-1 * vert_2[0][0], -1 * vert_2[0][1]]
            return [vert_1, vert_2, vert_3, vert_4]
        else:
            vert_4 = [-1 * vert_1[0][0], -1 * vert_1[0][1]]
            return [vert_1, vert_2, vert_3, vert_4]

    def __determinar_vertices_proximos(self, vertices):
        combinacao_vertices = combinations(range(len(vertices)), 2)

        distancia_vertices = []
        for i, j in combinacao_vertices:
            vertice_i, vertice_j = vertices[i][0], vertices[j][0]
            distancia = int(np.sqrt(((vertice_i[0] - vertice_j[0]) ** 2) + ((vertice_i[1] - vertice_j[1]) ** 2)))
            distancia_vertices.append((vertice_i, vertice_j, distancia))

        threshold = np.median([x[2] for x in distancia_vertices]) - 2 * np.std([x[2] for x in distancia_vertices])
        vertices_proximos = [x for x in distancia_vertices if x[2] < threshold]

        novos_vertices = []
        for vertice in vertices_proximos:
            vertice = [[vertice[0]], [vertice[1]]]
            novos_vertices.append(self.__encontrar_vertices(vertice, clusters=1))

        for v in vertices_proximos:
            vertices.remove([v[0]])
            vertices.remove([v[1]])

        for vertice in novos_vertices:
            vertices.append(vertice[0])

        return vertices

    @staticmethod
    def angulo_entre_pontos(vertices):
        vert_1, vert_2 = vertices[0][0], vertices[1][0]

        ang1 = np.arctan2(*vert_1[::-1])
        ang2 = np.arctan2(*vert_2[::-1])

        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    @staticmethod
    def __angulo_entre_retas(reta_1, reta_2):
        angulo_1 = np.arctan2(*reta_1[::-1])
        angulo_2 = np.arctan2(*reta_2[::-1])

        return np.rad2deg((angulo_1 - angulo_2) % (2 * np.pi))

    @staticmethod
    def __determinar_angulo_entre_retas(linha_1, linha_2):
        r1, theta1 = linha_1
        r2, theta2 = linha_2

        try:
            m1 = -(np.cos(theta1) / np.sin(theta1))
            m2 = -(np.cos(theta2) / np.sin(theta2))
            return abs(atan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)
        except ZeroDivisionError:
            return 180

    @staticmethod
    def __encontrar_vertices(intersecoes, clusters=4):
        x = np.array([[ponto[0][0], ponto[0][1]] for ponto in intersecoes])
        kmeans = KMeans(
            n_clusters=clusters,
            init='k-means++',
            max_iter=100,
            n_init=10,
            random_state=0
        ).fit(x)

        return [[centro.tolist()] for centro in kmeans.cluster_centers_]

    @staticmethod
    def __intersecao(linha_1, linha_2):
        r1, theta1 = linha_1
        r2, theta2 = linha_2

        a = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])

        b = np.array([[r1], [r2]])
        x0, y0 = np.linalg.solve(a, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

