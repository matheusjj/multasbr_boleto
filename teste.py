import cv2
import sys 
import extrator_info as ext

sys.setrecursionlimit(10 ** 9)

img = cv2.imread('imagens/folha_2.jpeg')
nome_janela = 'Teste'
t = ext.RoiAjustavel(img, nome_janela = nome_janela, 
                     lar_janela = img.shape[1], alt_janela = img.shape[0])
cv2.namedWindow(t.nome_janela)
cv2.setMouseCallback(nome_janela, ext.arrastarQuad, t)

while True:
    cv2.imshow(t.nome_janela, t.imagem)
    key = cv2.waitKey(1) and 0xFF
    
    if t.return_flag:
        break

cv2.destroyAllWindows()
