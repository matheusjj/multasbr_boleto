import imutils
import cv2 as cv2
import numpy as np
from itertools import combinations

from scipy import ndimage as ndi

from sklearn.cluster import KMeans
from skimage.filters import rank
from skimage.morphology import disk
from skimage.segmentation import watershed

from entidades.quadrilatero import Quadrilatero
from entidades.saturacao import Saturacao
from detectores.detector_texto import DetectorProjecao

import matplotlib.pyplot as plt
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

        if linhas is None:
            return Quadrilatero()

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

        if len(intersecoes) == 0:
            return Quadrilatero()

        vertices = self.encontrar_vertices(intersecoes, clusters=len(intersecoes) if len(intersecoes) < 4 else 4)
        vertices = self.__determinar_vertices_proximos(vertices)

        if vertices is None:
            return Quadrilatero()

        if len(vertices) <= 1:
            return Quadrilatero()

        elif len(vertices) == 2:
            vertices = self.completar_diagonal(vertices[0], vertices[1])

            if vertices is None:
                return Quadrilatero()
            else:
                return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))

        elif len(vertices) == 3:
            vertices = self.completar_vertices(vertices[0], vertices[1], vertices[2])

            if len(vertices) == 3:
                return Quadrilatero()

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
            te = [be[0] + delta_x, td[1]]
        elif td is None:
            delta_x = te[0] - be[0]
            td = [bd[0] + delta_x, te[1]]
        elif bd is None:
            delta_x = be[0] - te[0]
            bd = [td[0] + delta_x, be[1]]
        else:
            delta_x = bd[0] - td[0]
            be = [te[0] + delta_x, bd[1]]

        return [te, td, bd, be]

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
        self.ang_diagonal_inf = 35
        self.ang_diagonal_sup = 55

    def __call__(self, img) -> Quadrilatero:
        # Pré-processamento
        nivel_saturacao = self.avaliar_saturacao(img)

        if nivel_saturacao is Saturacao.BAIXISSIMA:
            cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(cinza, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            gx = cv2.convertScaleAbs(gx)

            detector = DetectorProjecao()
            pontos = detector(cv2.cvtColor(gx, cv2.COLOR_GRAY2BGR))

            return self.otimizar_vertices(np.array(pontos, dtype='uint16'), img)

        mean = cv2.pyrMeanShiftFiltering(img, 21, 51)
        lab = cv2.split(cv2.cvtColor(mean, cv2.COLOR_BGR2LAB))

        if nivel_saturacao is Saturacao.BAIXA:
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            # morph_l = cv2.morphologyEx(lab[0], cv2.MORPH_CLOSE, kernel)
            morph_l = ndi.binary_fill_holes(
                cv2.threshold(lab[0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            ).astype(int)
            morph_l = np.array(morph_l, dtype='uint8')
            # morph_l[morph_l == 1] = 255

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph_a = cv2.morphologyEx(lab[1], cv2.MORPH_ERODE, kernel)

            # morph_b = cv2.GaussianBlur(lab[2], (5, 5), 0)
            morph_b = ndi.binary_fill_holes(
                cv2.threshold(lab[2], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            ).astype(int)
            morph_b = np.array(morph_b, dtype='uint8')
            # morph_b[morph_b == 1] = 255

            l_somado = np.sum(morph_l) / (img.shape[0] * img.shape[1])
            b_somado = np.sum(morph_b) / (img.shape[0] * img.shape[1])

            if abs(b_somado - l_somado) >= 0.3:
                if l_somado <= 0.1:
                    morph_l = morph_b
                elif b_somado <= 0.1:
                    pass
                else:
                    lb_somado = np.ones(img.shape[:2], dtype='uint8')
                    lb_somado = cv2.bitwise_and(
                        lb_somado, lb_somado,
                        mask=cv2.bitwise_and(
                            np.array(morph_l, dtype='uint8'),
                            np.array(morph_b, dtype='uint8'))
                        )
                    lb_somado = np.sum(lb_somado) / (img.shape[0] * img.shape[1])

                    l_lb_razao, b_lb_razao = abs(l_somado / lb_somado - 1), \
                                             abs(b_somado / lb_somado - 1)
                    minimo = min([l_lb_razao, b_lb_razao])

                    if minimo == l_lb_razao:
                        morph_l = morph_l
                    else:
                        morph_l = morph_b
            else:
                morph_l = cv2.bitwise_and(morph_l, morph_b)

            # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
            # ax = axes.ravel()
            # ax[0].imshow(morph_l, cmap=plt.cm.gray)
            # ax[0].set_title("Morph L")
            # for a in ax:
            #     a.axis('off')
            # fig.tight_layout()
            # plt.savefig('Teste.png', format='png')
        elif nivel_saturacao is Saturacao.ALTA:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            morph_l = cv2.morphologyEx(lab[0], cv2.MORPH_CLOSE, kernel)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph_a = cv2.morphologyEx(lab[1], cv2.MORPH_ERODE, kernel)

            morph_b = cv2.GaussianBlur(lab[2], (5, 5), 0)

        # Segmentação de Fronteiras
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_l_grad = cv2.morphologyEx(morph_l, cv2.MORPH_GRADIENT, kernel)
        morph_a_grad = cv2.morphologyEx(morph_a, cv2.MORPH_GRADIENT, kernel)
        morph_b_grad = cv2.morphologyEx(morph_b, cv2.MORPH_GRADIENT, kernel)

        if nivel_saturacao is Saturacao.BAIXA:
            morph_grad = morph_l_grad
        elif nivel_saturacao is Saturacao.ALTA:
            morph_grad = morph_l_grad + morph_b_grad

        grad = cv2.morphologyEx(morph_grad, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Segmentação em regiões (Watershed)
        gradiente = rank.gradient(grad, disk(2))

        markers = rank.gradient(gradiente, disk(5)) < 10
        markers = ndi.label(markers)[0]

        labels = watershed(gradiente, markers)

        areas = thresh if \
            np.min(markers) == np.max(markers) else \
            np.array(
                (markers - np.min(markers)) / (np.max(markers) - np.min(markers)) * 255,
                dtype='uint8'
            )

        # Determinar retas
        retas = cv2.HoughLines(areas, 1, np.pi / 180, 140)

        if retas is None:
            return Quadrilatero()

        retas_verticais, retas_horizontais = self.determinar_retas(retas)
        retas_verticais, retas_horizontais = retas_verticais[:75], retas_horizontais[:75]

        # Determinar interseções
        intersecoes_gradiente = self.determinar_intersecoes(img, retas_verticais, retas_horizontais)

        if len(intersecoes_gradiente) == 0:
            return Quadrilatero()

        ponto_label = self.encontrar_vertices(intersecoes_gradiente)

        # cv2.circle(img, (int(ponto_label[0][0]), int(ponto_label[0][1])), 10, 255, -1)

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
        mascara_erode = cv2.morphologyEx(gradiente_sem_noise, cv2.MORPH_ERODE, kernel, iterations=7)
        mascara_erode = cv2.bitwise_and(grad, grad, mask=cv2.bitwise_not(mascara_erode))

        mascara_dilate = cv2.morphologyEx(gradiente_sem_noise, cv2.MORPH_DILATE, kernel, iterations=7)
        mascara_dilate = cv2.bitwise_and(mascara_erode, mascara_erode, mask=mascara_dilate)

        gradiente_sem_noise = cv2.threshold(mascara_dilate, 0, 255,
                                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), sharex=True, sharey=True)
        # ax = axes.ravel()
        #
        # ax[0].imshow(morph_l, cmap=plt.cm.gray)
        # ax[0].set_title("Morph L")
        #
        # ax[1].imshow(morph_a, cmap=plt.cm.gray)
        # ax[1].set_title("Morph A")
        #
        # ax[2].imshow(morph_b, cmap=plt.cm.gray)
        # ax[2].set_title("Morph B")
        #
        # ax[3].imshow(morph_l_grad, cmap=plt.cm.gray)
        # ax[3].set_title("Morph L Grad")
        #
        # ax[4].imshow(morph_a_grad, cmap=plt.cm.gray)
        # ax[4].set_title("Morph A Grad")
        #
        # ax[5].imshow(morph_b_grad, cmap=plt.cm.gray)
        # ax[5].set_title("Morph B Grad")
        #
        # ax[6].imshow(markers, cmap=plt.cm.nipy_spectral)
        # ax[6].set_title("Markers")
        #
        # ax[7].imshow(img, cmap=plt.cm.gray)
        # ax[7].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
        # ax[7].set_title("Labels")
        #
        # # ty = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # ty = cv2.GaussianBlur(ty, (7, 7), 0)
        # # ty = imutils.auto_canny(ty)
        # # b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, (28, 7))
        # ax[8].imshow(b, cmap=plt.cm.gray)
        # ax[8].set_title("Gradiente Final")
        #
        # for a in ax:
        #     a.axis('off')
        #
        # fig.tight_layout()
        # plt.savefig('Teste.png', format='png')

        # Determinar retas de grad sem áreas internas e externas
        retas = cv2.HoughLines(gradiente_sem_noise, 1, np.pi / 180, 90)

        if retas is None:
            vertices = self.encontrar_vertices(intersecoes_gradiente, clusters=4) if \
                len(intersecoes_gradiente) >=4 else \
                self.encontrar_vertices(intersecoes_gradiente, clusters=len(intersecoes_gradiente))
            vertices = self.determinar_vertices_proximos(vertices)
            return self.otimizar_vertices(vertices, img)

        retas_verticais, retas_horizontais = self.determinar_retas(retas)
        intersecoes_grad_sem_noise = self.determinar_intersecoes(img, retas_verticais,
                                                                 retas_horizontais)

        if len(intersecoes_grad_sem_noise) == 0:
            vertices = self.encontrar_vertices(intersecoes_gradiente, clusters=4)
            vertices = self.determinar_vertices_proximos(vertices)
            return self.otimizar_vertices(vertices, img)

        vertices_grad_sem_noise = \
            self.encontrar_vertices(intersecoes_grad_sem_noise, clusters=4) if\
            len(intersecoes_grad_sem_noise) >= 4 else intersecoes_grad_sem_noise

        vertices_grad_sem_noise = self.determinar_vertices_proximos(vertices_grad_sem_noise)
        return self.otimizar_vertices(vertices_grad_sem_noise, img)

    @staticmethod
    def avaliar_saturacao(img, thresh=60):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturacao = np.sum(hsv[:, :, 1]) / (hsv.shape[0] * hsv.shape[1])
        print(saturacao)

        if saturacao < 20:
            return Saturacao.BAIXISSIMA
        elif saturacao < thresh:
            return Saturacao.BAIXA
        else:
            return Saturacao.ALTA

    @staticmethod
    def filtrar_background(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        h_thresh = ndi.binary_fill_holes(
            cv2.threshold(
                hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]
        ).astype(int)
        b_thresh = ndi.binary_fill_holes(
            cv2.threshold(
                lab[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
        ).astype(int)

        v_thresh = ndi.binary_fill_holes(
            cv2.threshold(
                hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]
        ).astype(int)
        l_thresh = ndi.binary_fill_holes(
            cv2.threshold(
                lab[:, :, 0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]
        ).astype(int)

        intersecao = cv2.bitwise_and(h_thresh, b_thresh)
        uniao = cv2.bitwise_or(h_thresh, b_thresh)
        uniao = np.count_nonzero(uniao) if np.count_nonzero(uniao) != 0 else 1
        iou = np.count_nonzero(intersecao) / uniao
        iou_h_b = iou

        intersecao = cv2.bitwise_and(v_thresh, l_thresh)
        uniao = cv2.bitwise_or(v_thresh, l_thresh)
        uniao = np.count_nonzero(uniao) if np.count_nonzero(uniao) != 0 else 1
        iou = np.count_nonzero(intersecao) / uniao
        iou_v_l = iou

        if iou_h_b >= iou_v_l:
            mascara = cv2.bitwise_and(h_thresh, b_thresh)
        else:
            mascara = cv2.bitwise_and(v_thresh, l_thresh)

        return cv2.bitwise_and(img, img, mask=np.array(mascara, dtype='uint8'))

    def determinar_retas(self, retas):
        retas_verticais = []
        retas_horizontais = []

        for reta in retas:
            r, theta = reta[0][0], reta[0][1]
            if np.rad2deg(theta) <= self.angulo_reta_vertical_inf \
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

                if self.no_intervalo(ponto[0], img.shape[1]) \
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
        if vert_1[0] == vert_2[0] and vert_1[1] == vert_2[1]:
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

            if len(vertices) != 4:
                return Quadrilatero()
            return Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))
        else:
            quad = Quadrilatero(np.array(vertices, dtype='float32').reshape(4, 2))
            vertices = quad.retornar_vertices()

            if quad.e_concavo(0):
                vertices = self.completar_vertices(vertices[1], vertices[2], vertices[3])
            elif quad.e_concavo(1):
                vertices = self.completar_vertices(vertices[0], vertices[2], vertices[3])
            elif quad.e_concavo(2):
                vertices = self.completar_vertices(vertices[0], vertices[1], vertices[3])
            elif quad.e_concavo(3):
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

