import cv2
import imutils
import numpy as np

class Quadrilatero:
    def __init__(self) -> None:
        self.te = (0, 0)
        self.td = (0, 0)
        self.bd = (0, 0)
        self.be = (0, 0)

    def retornarPontos(self) -> tuple:
        return self.te, self.td, self.bd, self.be

    def eQuad(self) -> bool:
        return not (self.te == self.td or self.te == self.bd or self.te == self.be)

    def x_diff(self) -> tuple:
        return self.td[0] - self.te[0], self.bd[0] - self.td[0], self.be[0] - self.bd[0], self.te[0] - self.be[0]

    def y_diff(self) -> tuple:
        return self.td[1] - self.te[1], self.bd[1] - self.td[1], self.be[1] - self.bd[1], self.te[1] - self.be[1]

    def eColinear(self, ponto_id) -> bool:
        if ponto_id == 0:
            dist_te_td = int(np.sqrt(((self.td[0] - self.te[0]) ** 2) + ((self.td[1] - self.te[1]) ** 2)))
            dist_te_be = int(np.sqrt(((self.be[0] - self.te[0]) ** 2) + ((self.be[1] - self.te[1]) ** 2)))
            dist_be_td = int(np.sqrt(((self.td[0] - self.be[0]) ** 2) + ((self.td[1] - self.be[1]) ** 2)))
            return dist_te_td + dist_te_be == dist_be_td 
        elif ponto_id == 1:
            dist_td_bd = int(np.sqrt(((self.bd[0] - self.td[0]) ** 2) + ((self.bd[1] - self.td[1]) ** 2)))
            dist_td_te = int(np.sqrt(((self.te[0] - self.td[0]) ** 2) + ((self.te[1] - self.td[1]) ** 2)))
            dist_te_bd = int(np.sqrt(((self.bd[0] - self.te[0]) ** 2) + ((self.bd[1] - self.te[1]) ** 2)))
            # print(int(dist_td_bd), int(dist_td_te), int(dist_te_bd), int(dist_td_bd) + int(dist_td_te) == int(dist_te_bd))
            return dist_td_bd + dist_td_te == dist_te_bd
        elif ponto_id == 2:
            dist_bd_be = int(np.sqrt(((self.be[0] - self.bd[0]) ** 2) + ((self.be[1] - self.bd[1]) ** 2)))
            dist_bd_td = int(np.sqrt(((self.td[0] - self.bd[0]) ** 2) + ((self.td[1] - self.bd[1]) ** 2)))
            dist_td_be = int(np.sqrt(((self.be[0] - self.td[0]) ** 2) + ((self.be[1] - self.td[1]) ** 2)))
            return dist_bd_be + dist_bd_td == dist_td_be 
        else:
            dist_be_te = int(np.sqrt(((self.te[0] - self.be[0]) ** 2) + ((self.te[1] - self.be[1]) ** 2)))
            dist_be_bd = int(np.sqrt(((self.bd[0] - self.be[0]) ** 2) + ((self.bd[1] - self.be[1]) ** 2)))
            dist_bd_te = int(np.sqrt(((self.te[0] - self.bd[0]) ** 2) + ((self.te[1] - self.bd[1]) ** 2)))
            return dist_be_te + dist_be_bd == dist_bd_te

class RoiAjustavel:
    # Dimensões do canvas
    dimensoesCanvas = Quadrilatero()

    # Dimensões do ROI
    dimensoesRoi = Quadrilatero()

    # Imagem
    imagem = None

    # Nome da janela
    nome_janela = ''

    # Valores
    raio = 5
    ponto_id = None
    primeiro_ponto = (None, None)

    # Flags
    returnFlag = False
    ativo = False
    arrastar = False
    segurar = False

    def __init__(self, img, nome_janela, lar_janela, alt_janela) -> None:
        self.imagem = img
        self.nome_janela = nome_janela

        self.dimensoesCanvas.te = (0, 0)
        self.dimensoesCanvas.td = (lar_janela, 0)
        self.dimensoesCanvas.bd = (lar_janela, alt_janela)
        self.dimensoesCanvas.be = (0, alt_janela)

        self.dimensoesRoi.te = (0, 0)
        self.dimensoesRoi.td = (0, 0)
        self.dimensoesRoi.bd = (0, 0)
        self.dimensoesRoi.be = (0, 0)

def arrastarQuad(event, x, y, flags, quad_roi):
    if event == cv2.EVENT_LBUTTONDOWN:
        # cliqueEsquerdo(x, y, quad_roi)
        if quad_roi.ativo:
            quad_roi.ponto_id = determinarMarcador(x, y, quad_roi)

            if quad_roi.ponto_id == -1:
                return

            quad_roi.segurar = True
        else:
            quad_roi.primeiro_ponto = (x, y)
            quad_roi.arrastar = True
            quad_roi.ativo = True

    if event == cv2.EVENT_LBUTTONUP:
        # soltarClique(quad_roi)
        eQuad = quad_roi.dimensoesRoi.eQuad()
        quad_roi.arrastar = not eQuad
        quad_roi.ativo = eQuad

        quad_roi.segurar = False
        quad_roi.ponto_id = None

        if eQuad:
            atualizar_canvas(quad_roi)

    if quad_roi.ativo and event == cv2.EVENT_MOUSEMOVE:
        mouseMoveu(x, y, quad_roi)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # duploClique(quad_roi)
        quad_roi.return_flag = True
        cv2.destroyWindow(quad_roi.nome_janela)

    return

# def cliqueEsquerdo(mX, mY, quad_roi):
#     if quad_roi.ativo:
#         quad_roi.ponto_id = determinarMarcador(mX, mY, quad_roi)

#         if quad_roi.ponto_id == -1:
#             return

#         quad_roi.segurar = True
#     else:
#         quad_roi.primeiro_ponto = (mX, mY)
#         quad_roi.arrastar = True
#         quad_roi.ativo = True

# def soltarClique(quad_roi):
#     eQuad = quad_roi.dimensoesRoi.eQuad()
#     quad_roi.arrastar = not eQuad
#     quad_roi.ativo = eQuad

#     quad_roi.segurar = False
#     quad_roi.ponto_id = None

#     if eQuad:
#         atualizar_canvas(quad_roi)

def mouseMoveu(mX, mY, quad_roi):
    if mX > quad_roi.dimensoesCanvas.bd[0]:
        mX = quad_roi.dimensoesCanvas.bd[0] - 5
    elif mX < quad_roi.dimensoesCanvas.te[0]:
        mX = quad_roi.dimensoesCanvas.te[0] + 5

    if mY > quad_roi.dimensoesCanvas.bd[1]:
        mY = quad_roi.dimensoesCanvas.bd[1] - 5
    elif mY < quad_roi.dimensoesCanvas.te[1]:
        mY = quad_roi.dimensoesCanvas.te[1] + 5

    if quad_roi.arrastar:
        quad_roi.dimensoesRoi.te = quad_roi.primeiro_ponto
        quad_roi.dimensoesRoi.td = (mX, quad_roi.dimensoesRoi.te[1])
        quad_roi.dimensoesRoi.bd = (mX, mY)
        quad_roi.dimensoesRoi.be = (quad_roi.dimensoesRoi.te[0], mY)
        atualizar_canvas(quad_roi)

    elif quad_roi.segurar:
        if quad_roi.dimensoesRoi.eColinear(quad_roi.ponto_id):
            if quad_roi.ponto_id == 0:
                mX, mY = mX - 5, mY - 5
            elif quad_roi.ponto_id == 1:
                mX, mY = mX + 5, mY - 5
            elif quad_roi.ponto_id == 2:
                mX, mY = mX + 5, mY + 5
            elif quad_roi.ponto_id == 3:
                mX, mY = mX - 5, mY + 5
            quad_roi.segurar = False

        if quad_roi.ponto_id == 0:
            quad_roi.dimensoesRoi.te = (mX, mY)
        elif quad_roi.ponto_id == 1:
            quad_roi.dimensoesRoi.td = (mX, mY)
        elif quad_roi.ponto_id == 2:
            quad_roi.dimensoesRoi.bd = (mX, mY)
        elif quad_roi.ponto_id == 3:
            quad_roi.dimensoesRoi.be = (mX, mY)
        atualizar_canvas(quad_roi)

# def duploClique(quad_roi):
#     quad_roi.return_flag = True
#     cv2.destroyWindow(quad_roi.nome_janela)

def determinarMarcador(mX, mY, quad_roi):
    for idx, ponto in enumerate(quad_roi.dimensoesRoi.retornarPontos()):
        dist = np.sqrt(((ponto[0] - mX) ** 2) + ((ponto[1] - mY) ** 2))

        if dist <= quad_roi.raio:
            return idx

    return -1

def atualizar_canvas(quad_roi):
    temp = quad_roi.imagem.copy()
    pts = np.array(quad_roi.dimensoesRoi.retornarPontos())
    # (te, td, bd, be) = pts
    # min_x, max_x = np.min(pts, axis = 0)[0], np.max(pts, axis = 0)[0]
    # min_y, max_y = np.min(pts, axis = 0)[1], np.max(pts, axis = 0)[1]
    # mask = np.zeros(temp.shape[:2], dtype = 'uint8')
    # mask[te[1]:bd[1]+1, te[0]:bd[0]+1] = 1
    cv2.polylines(temp, [pts], True, (0, 255, 0), 2) 
    # temp = cv2.bitwise_and(temp, temp, mask = mask)
    # desenhaMarcadores(temp, quad_roi)

    (te, td, bd, be) = quad_roi.dimensoesRoi.retornarPontos()

    for ponto in (te, td, bd, be):
        cv2.circle(temp, (ponto[0], ponto[1]), quad_roi.raio, (0, 255, 0), -1)

    dx_te, dx_td, dx_bd, dx_be = quad_roi.dimensoesRoi.x_diff() 
    dy_te, dy_td, dy_bd, dy_be = quad_roi.dimensoesRoi.y_diff()

    cv2.line(temp, 
            (te[0] + int(dx_te / 2), te[1] + int(dy_te / 2)),
            (bd[0] + int(dx_bd / 2), bd[1] + int(dy_bd / 2)),
            (0, 255, 0), 2)
    cv2.line(temp, 
            (be[0] + int(dx_be / 2), be[1] + int(dy_be / 2)),
            (td[0] + int(dx_td / 2), td[1] + int(dy_td / 2)),
            (0, 255, 0), 2)
            
    cv2.imshow(quad_roi.nome_janela, temp)
    cv2.waitKey(0)

# def desenhaMarcadores(img, quad_roi):
#     (te, td, bd, be) = quad_roi.dimensoesRoi.retornarPontos()

#     for ponto in (te, td, bd, be):
#         cv2.circle(img, (ponto[0], ponto[1]), quad_roi.raio, (0, 255, 0), -1)

#     dx_te, dx_td, dx_bd, dx_be = quad_roi.dimensoesRoi.x_diff() 
#     dy_te, dy_td, dy_bd, dy_be = quad_roi.dimensoesRoi.y_diff()

#     cv2.line(img, 
#             (te[0] + int(dx_te / 2), te[1] + int(dy_te / 2)),
#             (bd[0] + int(dx_bd / 2), bd[1] + int(dy_bd / 2)),
#             (0, 255, 0), 2)
#     cv2.line(img, 
#             (be[0] + int(dx_be / 2), be[1] + int(dy_be / 2)),
#             (td[0] + int(dx_td / 2), td[1] + int(dy_td / 2)),
#             (0, 255, 0), 2)

def ordenar_pontos(pts):
    # Cria uma matriz onde serão armazenadas as coordenadas das vértices
    # do retângulo
    vertices_ret = np.zeros((4, 2), dtype = 'float32')
    
    # As maiores e menores somas correspondem, respectivamente, ao vértice
    # do topo mais à esquerda e na parte de baixo mais à direita
    s = pts.sum(axis = 1)
    vertices_ret[0] = pts[np.argmin(s)]
    vertices_ret[2] = pts[np.argmax(s)]

    # Os maiores e menores valores da diferenças entre a coordenada X e Y dos 
    # vértices correspondem, respectivamente, aos vértices da parte de baixa à
    # esquerda e no topo mais à direita
    diff = np.diff(pts, axis = 1)
    vertices_ret[1] = pts[np.argmin(diff)]
    vertices_ret[3] = pts[np.argmax(diff)]

    return vertices_ret

class ExtratorInfo:
    def __init__(self):
        return self

    def carregar_imagem(self, caminho):
        self.img = cv2.imread(caminho)
        self.pts = []

    def __tranformacao_quatro_vertices(self, img, pts):
        # Ordena os pontos da fronteira detectada
        self.pts = self.__ordenar_pontos(pts)
        (tl, tr, br, bl) = self.pts

        # Determina as larguras do retângulo e determina o maior
        largura_baixo = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        largura_cima = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        largura_max = max(int(largura_baixo), int(largura_cima))

        # Determina as alturas do retângulo e determina o maior
        altura_direita = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        altura_esquerda = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        altura_max = max(int(altura_direita), int(altura_esquerda))

        # Nova matriz para a qual o documento será transformado
        mtz_transf = np.array([
                            [0, 0],
                            [largura_max - 1, 0],
                            [largura_max - 1, altura_max - 1],
                            [0, altura_max - 1]], dtype = 'float32')

        M = cv2.getPerspectiveTransform(self.pts, mtz_transf)
        deslocado = cv2.warpPerspective(img, M, (largura_max, altura_max))

        return deslocado

    def __auto_canny(self, img, sigma = 0.33):
        media_intensidade = np.median(img)

        limite_menor = int(max(0, (1.0 - sigma) * media_intensidade))
        limite_maior = int(max(255, (1.0 + sigma) * media_intensidade))
        fronteiras = cv2.Canny(img, limite_menor, limite_maior)

        return fronteiras

    def __mouseCallback(self, event, x, y, flags, param):
        pass
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     self.pts.append((x, y))
        #     img_copy = self.img.copy()
        #     for pt in self.pts:
        #         (pX, pY) = pt
        #         cv2.circle(img_copy, (pX, pY), 10, (0, 255, 0), -1)
        #         if (len(pts) == 4):
        #             pts_np = np.array(pts, dtype = 'float32')
        #             warped = four_point_transform(im.copy(), pts_np)
        #             warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        #             # warped = cv2.GaussianBlur(warped, (3, 3), 0)
        #             cv2.imshow('Warped', imutils.resize(warped, height=600))
        #     cv2.imshow('Imagem', img_copy)

    def selecionar_manualmente(self):
        self.nome_janela = 'Selecao'
        cv2.imshow(self.nome_janela, self.img)
        
        if len(self.pts) > 0:
            img_copy = self.img.copy()
            for (pX, pY) in self.pts:
                cv2.circle(img_copy, (int(pX), int(pY)), 10, (0, 255, 0), -1)
            cv2.imshow(self.nome_janela, img_copy)
        # cv2.setMouseCallback(self.nome_janela, self.__mouseCallback)
        cv2.waitKey(0)

    def detectar_automaticamente(self, caminho):
        self.carregar_imagem(caminho)

        altura = 800
        (im_alt, im_lar) = self.img.shape[:2]
        ratio = im_alt / float(altura)

        img = imutils.resize(self.img, height = altura, inter = cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = self.__auto_canny(img)

        cnts = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

        for con in cnts:
            perimeter = cv2.arcLength(con, True)
            approx = cv2.approxPolyDP(con, 0.02 * perimeter, True)

            if len(approx) == 4:
                contorno = approx
                break

        if 'contorno' not in locals():
            self.selecionar_manualmente()
        else:
            # vertices = self.__tranformacao_quatro_vertices(self.img,
            #                                         contorno.reshape(4, 2) * ratio)
            self.pts = self.__ordenar_pontos(contorno.reshape(4, 2) * ratio)
            self.selecionar_manualmente()

    def teste(self, caminho):
        nome_janelo = 'o'
        img = cv2.imread(caminho)
        cv2.selectROI(nome_janelo, img, showCrosshair = False, fromCenter = True)
