import sys
import cv2
import extrator_info as ext

sys.setrecursionlimit(10 ** 9)

nome_janela = 'teste'
img = cv2.imread('imagens/folha_2.jpeg')
t = ext.RoiAjustavel(img, nome_janela, img.shape[1], img.shape[0])
cv2.namedWindow(nome_janela)
cv2.setMouseCallback(nome_janela, ext.arrastarQuad, t)

while True:
    cv2.imshow(nome_janela, t.imagem)
    key = cv2.waitKey(1) & 0xFF

    if t.returnFlag:
        break

cv2.destroyAllWindows()
