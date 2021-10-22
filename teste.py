import sys
import cv2
import numpy as np
import extrator_info as ext

def vazio():
    pass

sys.setrecursionlimit(10 ** 9)

key = 0
nome_janela = 'teste'
img = cv2.imread('imagens/folha_2.jpeg')
cv2.imshow(nome_janela, img)

roi = ext.RoiAjustavel(img, nome_janela, img.shape[1], img.shape[0])
cv2.setMouseCallback(nome_janela, ext.arrastarQuad, roi)

cv2.namedWindow(nome_janela)

while not key == 32:
    verde = (0, 255, 0)
    temp = roi.imagem.copy()
    pts = np.array(roi.dimensoesRoi.retornarPontos())
    cv2.polylines(temp, [pts], True, verde, 2) 

    (te, td, bd, be) = roi.dimensoesRoi.retornarPontos()

    for ponto in (te, td, bd, be):
        cv2.circle(temp, (ponto[0], ponto[1]), roi.raio, verde, -1)

    dx_te, dx_td, dx_bd, dx_be = roi.dimensoesRoi.x_diff() 
    dy_te, dy_td, dy_bd, dy_be = roi.dimensoesRoi.y_diff()

    cv2.line(temp, 
            (te[0] + int(dx_te / 2), te[1] + int(dy_te / 2)),
            (bd[0] + int(dx_bd / 2), bd[1] + int(dy_bd / 2)),
            verde, 2)
    cv2.line(temp, 
            (be[0] + int(dx_be / 2), be[1] + int(dy_be / 2)),
            (td[0] + int(dx_td / 2), td[1] + int(dy_td / 2)),
            verde, 2)

    cv2.imshow(nome_janela, temp)

    key = cv2.waitKey(30)
    if key == 'c' or key == 'C' or roi.returnFlag:
        break

cv2.setMouseCallback(nome_janela, vazio, None)
cv2.destroyAllWindows()
