import os
import cv2
import csv
import numpy as np
from extratores.extrator_documento import ExtratorDocumento
from detectores.detector_documento import DetectorDocumentoHough, DetectorLRDECustomizado
from pre_processadores.pre_processador import PreProcessador

# Testes angulo_entre_pontos


def test_angulo_entre_pontos_mesmo_ponto():
    vertices = [[1, 1], [1, 1]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_entre_pontos(vertices[0], vertices[1])
    assert angulo == 0


def test_angulo_entre_pontos_diagonal():
    vertices = [[-1, -1], [1, 1]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_entre_pontos(vertices[0], vertices[1])
    assert angulo == 180


def test_angulo_entre_pontos_diagonal_2():
    vertices = [[1, 1], [-1, -1]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_entre_pontos(vertices[0], vertices[1])
    assert angulo == 180


def test_angulo_entre_pontos_reta_horizontal():
    vertices = [[1, 1], [-1, 1]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_entre_pontos(vertices[0], vertices[1])
    assert angulo == 90


def test_angulo_entre_pontos_reta_vertical():
    vertices = [[1, 1], [1, -1]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_entre_pontos(vertices[0], vertices[1])
    assert angulo == 270


def test_angulo_entre_pontos_zero():
    vertices = [[1, 1], [1, 0]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_entre_pontos(vertices[0], vertices[1])
    assert angulo == 315


# Teste angulo_entre_retas


def test_angulo_entre_retas_mesma_reta():
    reta_1, reta_2 = [1, 0], [1, np.pi / 2]
    detector = DetectorDocumentoHough()
    angulo = detector.determinar_angulo_entre_retas(reta_1, reta_2)
    assert angulo == 90


def test_angulo_entre_retas_paralelas():
    reta_1, reta_2 = [1, 0], [2, 0]
    detector = DetectorDocumentoHough()
    angulo = detector.determinar_angulo_entre_retas(reta_1, reta_2)
    assert angulo == 0


def test_angulo_entre_retas_paralelas_opostas():
    reta_1, reta_2 = [1, 0], [2, np.pi]
    detector = DetectorDocumentoHough()
    angulo = detector.determinar_angulo_entre_retas(reta_1, reta_2)
    assert angulo == 180


# test encontrar_vertices


def test_encontrar_vertices_quatro_vertices():
    intersecoes = [[0, 0], [1, 1], [2, 2],
                   [98, 0], [99, 3], [97, 1],
                   [0, 99], [1, 98], [3, 97],
                   [99, 99], [97, 98], [100, 100]]
    detector = DetectorDocumentoHough()
    vertices = detector.encontrar_vertices(intersecoes)
    assert len(vertices) == 4
    assert vertices == [[98.0, 1.3333333333333357],
                        [98.66666666666666, 99.0],
                        [1.0, 1.000000000000007],
                        [1.3333333333333357, 98.0]]


def test_encontrar_vertices_tres_vertices():
    intersecoes = [[0, 0], [1, 1], [2, 2],
                   [98, 0], [99, 3], [97, 1],
                   [99, 99], [97, 98], [100, 100]]
    detector = DetectorDocumentoHough()
    vertices = detector.encontrar_vertices(intersecoes, clusters=3)
    assert len(vertices) == 3
    assert vertices == [[98.0, 1.3333333333333357],
                        [98.66666666666666, 99.0],
                        [1.0, 1.0]]


def test_encontrar_vertices_dois_vertices():
    intersecoes = [[0, 0], [1, 1], [2, 2],
                   [99, 99], [97, 98], [100, 100]]
    detector = DetectorDocumentoHough()
    vertices = detector.encontrar_vertices(intersecoes, clusters=2)
    assert len(vertices) == 2
    assert vertices == [[98.66666666666666, 99.0],
                        [1.000000000000007, 1.0]]


def test_encontrar_vertices_pontos_menores_clusters():
    intersecoes = [[0, 0], [99, 99]]
    detector = DetectorDocumentoHough()
    vertices = detector.encontrar_vertices(intersecoes)
    assert vertices is None


# test angulo_reta_entre_dois_pontos


def test_angulo_reta_inclinacao_negativa():
    vertices = [[100, 0], [0, 100]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_reta_entre_dois_pontos(vertices[0], vertices[1])
    assert angulo == -45


def test_angulo_reta_inclinacao_reta():
    vertices = [[100, 0], [100, 100]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_reta_entre_dois_pontos(vertices[0], vertices[1])
    assert angulo == 90


def test_angulo_reta_inclinacao_zero():
    vertices = [[0, 100], [100, 100]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_reta_entre_dois_pontos(vertices[0], vertices[1])
    assert angulo == 0


def test_angulo_reta_mesmo_ponto():
    vertices = [[100, 100], [100, 100]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_reta_entre_dois_pontos(vertices[0], vertices[1])
    assert angulo == 0


def test_angulo_reta_quadrantes_1_e_4():
    vertices = [[-78, 56], [86, -75]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_reta_entre_dois_pontos(vertices[0], vertices[1])
    assert angulo == -38.61717747362881


def test_angulo_reta_quadrantes_1_e_4_inv():
    vertices = [[86, -75], [-78, 56]]
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_reta_entre_dois_pontos(vertices[0], vertices[1])
    assert angulo == -38.61717747362881


def test_completar_diagonal_positiva():
    vertices = [[100, 0], [0, 100]]
    detector = DetectorDocumentoHough()
    vertices = detector.completar_diagonal(vertices[0], vertices[1])
    assert vertices == [[100, 0], [0, 100], [0, 0], [100, 100]]


def test_completar_diagonal_negativa():
    vertices = [[0, 100], [100, 0]]
    detector = DetectorDocumentoHough()
    vertices = detector.completar_diagonal(vertices[0], vertices[1])
    assert vertices == [[0, 100], [100, 0], [100, 100], [0, 0]]


def test_completar_diagonal_reta_horizontal():
    vertices = [[100, 0], [100, 100]]
    detector = DetectorDocumentoHough()
    vertices = detector.completar_diagonal(vertices[0], vertices[1])
    assert vertices is None


def test_completar_diagonal_reta_vertical():
    vertices = [[0, 100], [100, 100]]
    detector = DetectorDocumentoHough()
    vertices = detector.completar_diagonal(vertices[0], vertices[1])
    assert vertices is None


def test_completar_diagonal_mesmo_ponto():
    vertices = [[100, 100], [100, 100]]
    detector = DetectorDocumentoHough()
    vertices = detector.completar_diagonal(vertices[0], vertices[1])
    assert vertices is None


# test DetectorDocumentoHough


def test_detector_documento_hough_call():
    imagem = cv2.imread('../imagens/camera/1.jpg')

    pre_processador = PreProcessador()
    img, ratio = pre_processador(imagem)

    detector = DetectorDocumentoHough()
    detector(img)


# test DetectorLRDE


def test_detector_lrde():
    caminho_img = '/home/matheus/Documentos/Exemplo/'
    caminho_mascara_lrde = '/home/matheus/Documentos/Exemplo/Mascara_LRDE/'

    pre_processador = PreProcessador()
    detector = DetectorLRDECustomizado()
    extrator = ExtratorDocumento()

    for arquivo in os.listdir(caminho_img):
        if os.path.isfile(os.path.join(caminho_img, arquivo)):
            original = cv2.imread(caminho_img + arquivo)

            img, ratio = pre_processador(original)
            vertices = detector(img.copy())
            documento = extrator(img, vertices, original, ratio)
            quad = np.array([documento], dtype='int32')

            img_final = np.zeros(original.shape[:2])

            cv2.fillPoly(img_final, [quad], 255)
            cv2.imwrite(caminho_mascara_lrde + arquivo, img_final)

        # break


def test_processar_iou():
    caminho_auto = 'imagens/camera/mascara_doc_hough/'
    caminho_manual = 'imagens/camera/mascara_documento/'
    arquivo_output = 'Resultado_IoU_Hough'
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
