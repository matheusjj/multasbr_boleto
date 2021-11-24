import imutils
import numpy as np
import cv2.cv2 as cv2
from itertools import combinations

from scipy import stats
from scipy import ndimage as ndi

from sklearn.cluster import KMeans
from skimage.filters import rank
from skimage.morphology import disk
from skimage.segmentation import watershed

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

    # def adjust_gamma(image, gamma=1.0):
    #     inv_gamma = 1.0 / gamma
    #     table = np.array([((i / 255.0) ** inv_gamma) * 255
    #                       for i in np.arange(0, 256)]).astype("uint8")
    #     return cv2.LUT(image, table)
    #
    # # yuv = cv2.split(cv2.cvtColor(imagem, cv2.COLOR_BGR2YUV))
    # # t = cv2.equalizeHist(yuv[0])
    #
    # # eq = cv2.merge([t, yuv[1], yuv[2]])
    # # yuv = cv2.cvtColor(eq, cv2.COLOR_YUV2BGR)
    # # gamma = adjust_gamma(yuv, 0.9)
    # # hsv = cv2.cvtColor(gamma, cv2.COLOR_BGR2HSV)
    # # hsv = cv2.cvtColor(yuv, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    #
    # lower = np.array([0, 0, 220])
    # upper = np.array([180, 100, 255])
    #
    # mask = cv2.inRange(hsv, lower, upper)
    # result = cv2.bitwise_and(imagem, imagem, mask=mask)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # result = cv2.morphologyEx(result, cv2.MORPH_ERODE, kernel)
    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=20)
    #
    # # thresh = cv2.threshold(cv2.split(result)[2], 0, 255, cv2.THRESH_OTSU)[1]
    # thresh = cv2.threshold(cv2.split(result)[2], 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
    #
    # cv2.imshow('Mask', thresh)
    # cv2.waitKey(0)
    #
    # return imutils.auto_canny(thresh, sigma=0.5)


class DetectorDocumentoCanny:
    def __init__(self) -> None:
        pass

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
    def __init__(self, angulo_agudo_lim=80.0, angulo_obtuso_lim=100.0, votos_hough=30,
                 ang_diagonal_inf=40.0, ang_diagonal_sup=50.0):
        self.angulo_agudo = angulo_agudo_lim
        self.angulo_obtuso = angulo_obtuso_lim
        self.votos_hough = votos_hough
        self.ang_diagonal_inf = ang_diagonal_inf
        self.ang_diagonal_sup = ang_diagonal_sup

    def __call__(self, imagem):
        canny = pre_processamento_deteccao(imagem)
        linhas = cv2.HoughLines(canny, 1, np.pi / 180, self.votos_hough)

        cv2.imshow('A', canny)
        cv2.waitKey(0)

        if linhas is None:
            return None

        intersecoes = []
        linhas_agrupadas = combinations(range(len(linhas)), 2)
        x_no_intervalo = lambda x: 0 <= x <= imagem.shape[1]
        y_no_intervalo = lambda y: 0 <= y <= imagem.shape[0]

        for i, j in linhas_agrupadas:
            linha_i, linha_j = linhas[i][0], linhas[j][0]

            if self.angulo_agudo < self.determinar_angulo_entre_retas(linha_i, linha_j) < self.angulo_obtuso:
                int_point = self.__intersecao(linha_i, linha_j)

                if x_no_intervalo(int_point[0]) and y_no_intervalo(int_point[1]):
                    intersecoes.append(int_point)

        vertices = self.encontrar_vertices(intersecoes, clusters=len(intersecoes) if len(intersecoes) < 4 else 4)
        for vertice in vertices:
            vertice = vertice
            print(vertice)
            cv2.circle(imagem, (int(vertice[0]), int(vertice[1])), 10, 255, -1)

        cv2.imshow('A', imagem)
        cv2.waitKey(0)

        vertices = self.__determinar_vertices_proximos(vertices)

        if vertices is None:
            return None

        if len(vertices) <= 1:
            return Quadrilatero(np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype='float32').reshape(4, 2))
        elif len(vertices) == 2:
            vertices = self.completar_diagonal(vertices[0], vertices[1])

            if vertices is None:
                return Quadrilatero(np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype='float32').reshape(4, 2))
            else:
                return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))

        elif len(vertices) == 3:
            print(vertices)
            vertices = self.completar_vertices(vertices[0], vertices[1], vertices[2])
            print(vertices)

            for vertice in vertices:
                vertice = vertice
                cv2.circle(imagem, (int(vertice[0]), int(vertice[1])), 10, 255, -1)

            cv2.imshow('A', imagem)
            cv2.waitKey(0)

            return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))
        else:
            return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))

    def completar_diagonal(self, vert_1, vert_2):
        angulo_entre_pontos = self.angulo_reta_entre_dois_pontos(vert_1, vert_2)

        if self.ang_diagonal_inf <= abs(angulo_entre_pontos) <= self.ang_diagonal_sup:
            if angulo_entre_pontos > 0:
                vert_3 = [vert_1[0], vert_2[1]]
                vert_4 = [vert_2[0], vert_1[1]]
                return [vert_1, vert_2, vert_3, vert_4]
            else:
                vert_3 = [vert_2[0], vert_1[1]]
                vert_4 = [vert_1[0], vert_2[1]]
                return [vert_1, vert_2, vert_3, vert_4]
        else:
            return None

    @staticmethod
    def angulo_reta_entre_dois_pontos(vert_1, vert_2):
        if vert_1 == vert_2:
            return 0

        if vert_1[0] == vert_2[0]:
            ang = 90
        else:
            ang = np.rad2deg(np.arctan((vert_2[1] - vert_1[1]) / (vert_2[0] - vert_1[0])))

        return ang

    def completar_vertices(self, vert_1, vert_2, vert_3):
        ang_vert1_vert2 = self.angulo_reta_entre_dois_pontos(vert_1, vert_2)
        ang_vert1_vert3 = self.angulo_reta_entre_dois_pontos(vert_1, vert_3)
        # ang_vert3_vert2 = self.angulo_reta_entre_dois_pontos(vert_3, vert_2)
        #
        # ang_vert2_vert3 = self.angulo_reta_entre_dois_pontos(vert_2, vert_3)
        # ang_vert2_vert1 = self.angulo_reta_entre_dois_pontos(vert_2, vert_1)

        if self.ang_diagonal_inf <= abs(ang_vert1_vert2) <= self.ang_diagonal_sup:
            vert_4 = [vert_1[0] + vert_2[0] - vert_3[0], vert_1[1] + vert_2[1] - vert_3[1]]
            return [vert_1, vert_2, vert_3, vert_4]
        elif self.ang_diagonal_inf <= abs(ang_vert1_vert3) <= self.ang_diagonal_sup:
            vert_4 = [vert_1[0] + vert_3[0] - vert_2[0], vert_1[1] + vert_3[1] - vert_2[1]]
            return [vert_1, vert_2, vert_3, vert_4]
        else:
            # distancias = []
            # distancias.append(int(np.sqrt(((vert_1[0] - 0) ** 2) + ((vert_1[1] - 0) ** 2))))
            # distancias.append(int(np.sqrt(((vert_2[0] - 0) ** 2) + ((vert_2[1] - 0) ** 2))))
            # distancias.append(int(np.sqrt(((vert_3[0] - 0) ** 2) + ((vert_3[1] - 0) ** 2))))
            #
            # if min(distancias) == distancias[0]:
            #     vert_4 = [vert_2[0] + (vert_1[0] - vert_3[0]), vert_3[1] + (vert_1[1] - vert_2[1])]
            # else:
            #     vert_4 = [vert_3[0] + (vert_1[0] - vert_2[0]), vert_2[1] + (vert_1[1] - vert_3[1])]


            vert_4 = []

            return [vert_1, vert_2, vert_3, vert_4]

    def __determinar_vertices_proximos(self, vertices):
        combinacao_vertices = combinations(range(len(vertices)), 2)

        distancia_vertices = []
        for i, j in combinacao_vertices:
            vertice_i, vertice_j = vertices[i], vertices[j]
            distancia = int(np.sqrt(((vertice_i[0] - vertice_j[0]) ** 2) + ((vertice_i[1] - vertice_j[1]) ** 2)))
            distancia_vertices.append((vertice_i, vertice_j, distancia))

        threshold = np.median([x[2] for x in distancia_vertices]) - 2 * np.std([x[2] for x in distancia_vertices])
        vertices_proximos = [x for x in distancia_vertices if x[2] < threshold]

        novos_vertices = []
        for vertice in vertices_proximos:
            vertice = [vertice[0], vertice[1]]
            novos_vertices.append(self.encontrar_vertices(vertice, clusters=1))

        vertices_final = vertices.copy()
        for v in vertices_proximos:
            vertices_final.remove(v[0])
            vertices_final.remove(v[1])

        for vertice in novos_vertices:
            vertices_final.append(vertice[0])

        return vertices_final

    def determinar_angulo_entre_retas(self, reta_1, reta_2):
        r1, teta_1 = reta_1
        r2, teta_2 = reta_2

        vertices = [[r1 * np.cos(teta_1), r1 * np.sin(teta_1)],
                    [r2 * np.cos(teta_2), r2 * np.sin(teta_2)]]

        return self.angulo_entre_pontos(vertices[0], vertices[1])

    @staticmethod
    def angulo_entre_pontos(vert_1, vert_2):
        ang1 = np.arctan2(*vert_1[::-1])
        ang2 = np.arctan2(*vert_2[::-1])

        return np.rad2deg((ang2 - ang1) % (2 * np.pi))

    @staticmethod
    def encontrar_vertices(intersecoes, clusters=4):
        if len(intersecoes) < clusters:
            return None

        x = np.array([[ponto[0], ponto[1]] for ponto in intersecoes])
        kmeans = KMeans(
            n_clusters=clusters,
            init='k-means++',
            max_iter=100,
            n_init=10,
            random_state=0
        ).fit(x)

        return [centro.tolist() for centro in kmeans.cluster_centers_]

    @staticmethod
    def __intersecao(linha_1, linha_2):
        r1, teta_1 = linha_1
        r2, teta_2 = linha_2

        a = np.array([
            [np.cos(teta_1), np.sin(teta_1)],
            [np.cos(teta_2), np.sin(teta_2)]
        ])

        b = np.array([[r1], [r2]])
        x0, y0 = np.linalg.solve(a, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]


class DetectorLRDECustomizado:
    def __init__(self) -> None:
        self.angulo_reta_vertical_inf = 45
        self.angulo_reta_vertical_sup = 135
        self.regiao_label = 20
        self.ang_diagonal_inf = 40
        self.ang_diagonal_sup = 50

    def determinar_retas(self, retas):
        retas_verticais = []
        retas_horizontais = []

        for reta in retas:
            r, theta = reta[0][0], reta[0][1]
            if np.rad2deg(theta) <= self.angulo_reta_vertical_inf\
                    or np.rad2deg(theta) >= self.angulo_reta_vertical_sup:
                retas_verticais.append([r, theta])
            else:
                retas_horizontais.append([r, theta])

        return retas_verticais, retas_horizontais

    @staticmethod
    def intersecao(linha_1, linha_2):
        r1, teta_1 = linha_1
        r2, teta_2 = linha_2

        a = np.array([
            [np.cos(teta_1), np.sin(teta_1)],
            [np.cos(teta_2), np.sin(teta_2)]
        ])

        b = np.array([[r1], [r2]])
        x0, y0 = np.linalg.solve(a, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]

    @staticmethod
    def no_intervalo(x, tamanho):
        return 0 <= x <= tamanho

    def determinar_intersecoes(self, img, retas_verticais, retas_horizontais):
        intersecoes = []

        for vertical in retas_verticais:
            for horizontal in retas_horizontais:
                ponto = self.intersecao(vertical, horizontal)

                if self.no_intervalo(ponto[0], img.shape[1])\
                        and self.no_intervalo(ponto[1], img.shape[0]):
                    intersecoes.append(ponto)

        return intersecoes

    @staticmethod
    def encontrar_vertices(intersecoes, clusters=1):
        x = np.array([[ponto[0], ponto[1]] for ponto in intersecoes])
        kmeans = KMeans(
            n_clusters=clusters,
            init='k-means++',
            max_iter=100,
            n_init=10,
            random_state=0
        ).fit(x)

        return [centro.tolist() for centro in kmeans.cluster_centers_]

    @staticmethod
    def angulo_reta_entre_dois_pontos(vert_1, vert_2):
        if vert_1 == vert_2:
            return 0

        if vert_1[0] == vert_2[0]:
            ang = 90
        else:
            ang = np.rad2deg(np.arctan((vert_2[1] - vert_1[1]) / (vert_2[0] - vert_1[0])))

        return ang

    def completar_diagonal(self, vert_1, vert_2):
        angulo_entre_pontos = self.angulo_reta_entre_dois_pontos(vert_1, vert_2)

        if self.ang_diagonal_inf <= abs(angulo_entre_pontos) <= self.ang_diagonal_sup:
            if angulo_entre_pontos > 0:
                vert_3 = [vert_1[0], vert_2[1]]
                vert_4 = [vert_2[0], vert_1[1]]
                return [vert_1, vert_2, vert_3, vert_4]
            else:
                vert_3 = [vert_2[0], vert_1[1]]
                vert_4 = [vert_1[0], vert_2[1]]
                return [vert_1, vert_2, vert_3, vert_4]
        else:
            return None

    @staticmethod
    def completar_vertices(vert_1, vert_2, vert_3):
        t = np.array([vert_1, vert_2, vert_3])
        media_coord = np.average(t, axis=0)

        pontos_esq = []
        pontos_dir = []
        for ponto in [vert_1, vert_2, vert_3]:
            if ponto[0] <= media_coord[0]:
                pontos_esq.append(ponto)
            else:
                pontos_dir.append(ponto)

        te, td, bd, be = None, None, None, None

        for ponto in pontos_esq:
            if ponto[1] <= media_coord[1]:
                te = ponto
            else:
                be = ponto

        for ponto in pontos_dir:
            if ponto[1] <= media_coord[1]:
                td = ponto
            else:
                bd = ponto

        if sum(x is None for x in [te, td, bd, be]) > 1:
            return [vert_1, vert_2, vert_3]

        if te is None:
            delta_x = td[0] - bd[0]
            # delta_x = 0
            # delta_y = be[1] - bd[1]
            # te = [be[0] + delta_x, td[1] + delta_y]
            te = [be[0] + delta_x, td[1]]
        elif td is None:
            delta_x = te[0] - be[0]
            # delta_x = 0
            # delta_y = bd[1] - be[1]
            # td = [bd[0] + delta_x, te[1] + delta_y]
            td = [bd[0] + delta_x, te[1]]
        elif bd is None:
            delta_x = be[0] - te[0]
            # delta_x = 0
            # delta_y = td[1] - te[1]
            # bd = [td[0] + delta_x, be[1] + delta_y]
            bd = [td[0] + delta_x, be[1]]
        else:
            delta_x = bd[0] - td[0]
            # delta_x = 0
            # delta_y = te[1] - td[1]
            # be = [te[0] + delta_x, bd[1] + delta_y]
            be = [te[0] + delta_x, bd[1]]

        return [te, td, bd, be]

    @staticmethod
    def corrigir_vertices(vertices, img):
        verts = []

        for ponto in vertices:
            p = ponto
            if ponto[0] < 0:
                p[0] = 0
            elif ponto[0] > img.shape[1]:
                p[0] = img.shape[1]

            if ponto[1] < 0:
                p[1] = 0
            elif ponto[1] > img.shape[0]:
                p[1] = img.shape[0]

            verts.append(p)

        return verts

    def otimizar_vertices(self, vertices, img):
        if len(vertices) == 1:
            return Quadrilatero()
        elif len(vertices) == 2:
            vertices = self.completar_diagonal(vertices[0], vertices[1])

            if vertices is None:
                return Quadrilatero()
            else:
                return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))
        elif len(vertices) == 3:
            vertices = self.completar_vertices(vertices[0], vertices[1], vertices[2])
            vertices = self.corrigir_vertices(vertices, img)

            return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))
        else:
            quad = Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))
            vertices = quad.retornar_vertices()

            if quad.e_colinear(0):
                vertices = self.completar_vertices(vertices[1], vertices[2], vertices[3])
            elif quad.e_colinear(1):
                vertices = self.completar_vertices(vertices[0], vertices[2], vertices[3])
            elif quad.e_colinear(2):
                vertices = self.completar_vertices(vertices[0], vertices[1], vertices[3])
            elif quad.e_colinear(3):
                vertices = self.completar_vertices(vertices[0], vertices[1], vertices[2])

            vertices = self.corrigir_vertices(vertices, img)
            if len(vertices) != 4:
                return Quadrilatero()

            return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))

    def determinar_vertices_proximos(self, vertices):
        combinacao_vertices = combinations(range(len(vertices)), 2)

        distancia_vertices = []
        for i, j in combinacao_vertices:
            vertice_i, vertice_j = vertices[i], vertices[j]
            distancia = int(np.sqrt(((vertice_i[0] - vertice_j[0]) ** 2) + ((vertice_i[1] - vertice_j[1]) ** 2)))
            distancia_vertices.append((vertice_i, vertice_j, distancia))

        threshold = np.median([x[2] for x in distancia_vertices]) - 2 * np.std([x[2] for x in distancia_vertices])
        vertices_proximos = [x for x in distancia_vertices if x[2] < threshold]

        novos_vertices = []
        for vertice in vertices_proximos:
            vertice = [vertice[0], vertice[1]]
            novos_vertices.append(self.encontrar_vertices(vertice, clusters=1))

        vertices_final = vertices.copy()
        for v in vertices_proximos:
            vertices_final.remove(v[0])
            vertices_final.remove(v[1])

        for vertice in novos_vertices:
            vertices_final.append(vertice[0])

        return vertices_final

    def __call__(self, img) -> Quadrilatero:
        # Pré-processamento
        mean = cv2.pyrMeanShiftFiltering(img, 21, 51)
        lab = cv2.split(cv2.cvtColor(mean, cv2.COLOR_BGR2LAB))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        morph_l = cv2.morphologyEx(lab[0], cv2.MORPH_CLOSE, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_a = cv2.morphologyEx(lab[1], cv2.MORPH_ERODE, kernel)

        # Segmentação de Fronteiras
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_l_grad = cv2.morphologyEx(morph_l, cv2.MORPH_GRADIENT, kernel)
        morph_a_grad = cv2.morphologyEx(morph_a, cv2.MORPH_GRADIENT, kernel)
        morph_b_grad = cv2.morphologyEx(lab[2], cv2.MORPH_GRADIENT, kernel)

        morph_grad = morph_l_grad + morph_a_grad + morph_b_grad
        grad = cv2.morphologyEx(morph_grad, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Determinar retas
        retas = cv2.HoughLines(thresh, 1, np.pi / 180, 90)

        if retas is None:
            return Quadrilatero()

        retas_verticais, retas_horizontais = self.determinar_retas(retas)

        # Determinar interseções
        intersecoes_gradiente = self.determinar_intersecoes(img, retas_verticais,
                                                            retas_horizontais)

        if len(intersecoes_gradiente) == 0:
            return Quadrilatero()

        ponto_label = self.encontrar_vertices(intersecoes_gradiente)

        # Segmentação em regiões (Watershed)
        markers = rank.gradient(grad, disk(5)) < 10
        markers = ndi.label(markers)[0]
        gradiente = rank.gradient(grad, disk(2))
        labels = watershed(gradiente, markers)

        label_documento = np.unique(labels[int(ponto_label[0][1]) - self.regiao_label:
                                           int(ponto_label[0][1]) + self.regiao_label,
                                           int(ponto_label[0][0]) - self.regiao_label:
                                           int(ponto_label[0][0]) + self.regiao_label]
                                    .flatten())

        labels_relevantes = []
        for idx, num in enumerate(label_documento):
            labels_relevantes.append([idx, np.count_nonzero(labels == num)])
        labels_relevantes.sort(key=lambda n: n[1], reverse=True)

        # Máscaras retirar regiões dentro e fora do documento
        gradiente_sem_noise = np.zeros(img.shape[:2], dtype='uint8')
        for label in np.unique(labels):
            if label != label_documento[labels_relevantes[0][0]]:
                continue

            mask = np.zeros(img.shape[:2], dtype="uint8")
            mask[labels == label] = 255

            cnts, hier = cv2.findContours(mask.copy(),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gradiente_sem_noise, cnts, -1, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mascara_erode = cv2.morphologyEx(gradiente_sem_noise,
                                         cv2.MORPH_ERODE, kernel, iterations=7)
        mascara_erode = cv2.bitwise_and(grad, grad,
                                        mask=cv2.bitwise_not(mascara_erode))

        mascara_dilate = cv2.morphologyEx(gradiente_sem_noise,
                                          cv2.MORPH_DILATE, kernel, iterations=7)
        mascara_dilate = cv2.bitwise_and(mascara_erode, mascara_erode,
                                         mask=mascara_dilate)

        gradiente_sem_noise = cv2.threshold(mascara_dilate, 0, 255,
                                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Determinar retas de grad sem áreas internas e externas
        retas = cv2.HoughLines(gradiente_sem_noise, 1, np.pi / 180, 90)

        if retas is None:
            vertices = self.encontrar_vertices(intersecoes_gradiente, clusters=4)
            vertices = self.determinar_vertices_proximos(vertices)
            return self.otimizar_vertices(vertices, img)

        retas_verticais, retas_horizontais = self.determinar_retas(retas)
        intersecoes_grad_sem_noise = self.determinar_intersecoes(img, retas_verticais,
                                                                 retas_horizontais)

        if len(intersecoes_grad_sem_noise) == 0:
            vertices = self.encontrar_vertices(intersecoes_gradiente, clusters=4)
            vertices = self.determinar_vertices_proximos(vertices)
            return self.otimizar_vertices(vertices, img)

        vertices_grad_sem_noise = self.encontrar_vertices(intersecoes_grad_sem_noise,
                                                          clusters=4)
        vertices_grad_sem_noise = self.determinar_vertices_proximos(vertices_grad_sem_noise)
        return self.otimizar_vertices(vertices_grad_sem_noise, img)
