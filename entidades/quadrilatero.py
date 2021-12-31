import numpy as np
from itertools import combinations


class Quadrilatero:
    def __init__(self, vertices=None) -> None:
        if vertices is None:
            self.te = [0, 0]
            self.td = [0, 0]
            self.bd = [0, 0]
            self.be = [0, 0]
        else:
            vertices = self.ordenar_pontos(vertices)
            self.atualizar_vertices(vertices)

    def atualizar_vertices(self, vertices) -> None:
        self.te = vertices[0]
        self.td = vertices[1]
        self.bd = vertices[2]
        self.be = vertices[3]

    def e_concavo(self, ponto_id) -> bool:
        if ponto_id == 0:
            dist_te_td = int(np.sqrt(((self.td[0] - self.te[0]) ** 2) + ((self.td[1] - self.te[1]) ** 2)))
            dist_te_be = int(np.sqrt(((self.be[0] - self.te[0]) ** 2) + ((self.be[1] - self.te[1]) ** 2)))
            dist_be_td = int(np.sqrt(((self.td[0] - self.be[0]) ** 2) + ((self.td[1] - self.be[1]) ** 2)))
            return dist_te_td + dist_te_be <= dist_be_td
        elif ponto_id == 1:
            dist_td_bd = int(np.sqrt(((self.bd[0] - self.td[0]) ** 2) + ((self.bd[1] - self.td[1]) ** 2)))
            dist_td_te = int(np.sqrt(((self.te[0] - self.td[0]) ** 2) + ((self.te[1] - self.td[1]) ** 2)))
            dist_te_bd = int(np.sqrt(((self.bd[0] - self.te[0]) ** 2) + ((self.bd[1] - self.te[1]) ** 2)))
            return dist_td_bd + dist_td_te <= dist_te_bd
        elif ponto_id == 2:
            dist_bd_be = int(np.sqrt(((self.be[0] - self.bd[0]) ** 2) + ((self.be[1] - self.bd[1]) ** 2)))
            dist_bd_td = int(np.sqrt(((self.td[0] - self.bd[0]) ** 2) + ((self.td[1] - self.bd[1]) ** 2)))
            dist_td_be = int(np.sqrt(((self.be[0] - self.td[0]) ** 2) + ((self.be[1] - self.td[1]) ** 2)))
            return dist_bd_be + dist_bd_td <= dist_td_be
        else:
            dist_be_te = int(np.sqrt(((self.te[0] - self.be[0]) ** 2) + ((self.te[1] - self.be[1]) ** 2)))
            dist_be_bd = int(np.sqrt(((self.bd[0] - self.be[0]) ** 2) + ((self.bd[1] - self.be[1]) ** 2)))
            dist_bd_te = int(np.sqrt(((self.te[0] - self.bd[0]) ** 2) + ((self.te[1] - self.bd[1]) ** 2)))
            return dist_be_te + dist_be_bd <= dist_bd_te

    def e_quad(self) -> bool:
        combinacoes = combinations(range(4), 2)
        verts = [self.te, self.td, self.bd, self.be]
        for i, j in combinacoes:
            vert_i, vert_j = verts[i], verts[j]

            if vert_i[0] == vert_j[0] and vert_i[1] == vert_j[1]:
                return False

        return True

    def e_zerado(self) -> bool:
        return np.sum(self.retornar_vertices()) == 0

    def retornar_vertices(self) -> tuple:
        vertices = self.ordenar_pontos(np.array([self.te, self.td, self.bd, self.be]).reshape(4, 2))
        self.atualizar_vertices(vertices)

        return np.array([self.te, self.td, self.bd, self.be], dtype='float32').reshape(4, 2)

    # TODO: Tentar usar np diff aqui
    def x_diff(self) -> tuple:
        return self.td[0] - self.te[0], self.bd[0] - self.td[0], self.be[0] - self.bd[0], self.te[0] - self.be[0]

    def y_diff(self) -> tuple:
        return self.td[1] - self.te[1], self.bd[1] - self.td[1], self.be[1] - self.bd[1], self.te[1] - self.be[1]

    @staticmethod
    def ordenar_pontos(vertices):
        vertices_ret = np.zeros((4, 2), dtype='float32')

        if np.sum(vertices) == 0 or vertices.shape[0] != 4:
            return vertices_ret

        lim_vertical = 0
        for n in range(int(np.min(vertices, axis=0)[1]), int(np.max(vertices, axis=0)[1])):
            if np.count_nonzero(vertices[:, 1] <= n) >= 2:
                lim_vertical = n
                break

        vertices_superiores = [vertices[idx] for idx in
                               set(np.where(vertices[:, 1] <= lim_vertical)[0][:2])]
        vertices_inferiores = [vertices[idx] for idx in
                               set(range(0, len(vertices))) - set(np.where(vertices[:, 1] <= lim_vertical)[0][:2])]

        vertices_superiores.sort(key=lambda x: x[0])
        vertices_inferiores.sort(key=lambda x: x[0])

        vertices_ret[0] = vertices_superiores[0] if vertices_superiores[0] is not None else [0, 0]
        vertices_ret[1] = vertices_superiores[1] if vertices_superiores[1] is not None else [0, 0]
        vertices_ret[2] = vertices_inferiores[1] if vertices_inferiores[1] is not None else [0, 0]
        vertices_ret[3] = vertices_inferiores[0] if vertices_inferiores[0] is not None else [0, 0]

        return vertices_ret
