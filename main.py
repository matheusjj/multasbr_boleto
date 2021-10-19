import os
import cv2
import imutils
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from skimage.filters import threshold_local

def order_points(pts):
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

def four_point_transform(img, pts):
    retangulo = order_points(pts)
    (tl, tr, br, bl) = retangulo

    largura_baixo = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    largura_cima = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    largura_max = max(int(largura_baixo), int(largura_cima))

    altura_direita = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    altura_esquerda = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    altura_max = max(int(altura_direita), int(altura_esquerda))

    nova_dimensao = np.array([
                            [0, 0],
                            [largura_max - 1, 0],
                            [largura_max - 1, altura_max - 1],
                            [0, altura_max - 1]], dtype = 'float32')

    M = cv2.getPerspectiveTransform(retangulo, nova_dimensao)
    deslocado = cv2.warpPerspective(img, M, (largura_max, altura_max))

    return deslocado

def pdf_para_imagem(caminho):
    pages = convert_from_path(caminho, 500) 
    out_img = os.path.basename(caminho)
    out_img = os.path.splitext(out_img)[0]
    out_img = '{}.png'.format(out_img)
    caminho_sp = os.path.split(caminho)

    if len(caminho_sp) > 1:
        for page in pages:
            page.save('{}/{}'.format(caminho_sp[0], out_img))
    else:
        for page in pages:
            page.save('{}'.format(out_img))

def auto_canny(imagem, sigma = 0.33):

    media_intensidade = np.median(imagem)

    limite_menor = int(max(0, (1.0 - sigma) * media_intensidade))
    limite_maior = int(max(255, (1.0 + sigma) * media_intensidade))
    fronteiras = cv2.Canny(imagem, limite_menor, limite_maior)

    return fronteiras

def selecionar_documento(img):
    altura = 800
    orig = img.copy()
    proporcao = img.shape[0] / float(altura)

    img = imutils.resize(img, height = altura, inter = cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(cinza, (7, 7), 0)
    fronteira = auto_canny(blurred) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fronteira = cv2.morphologyEx(fronteira, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Original', img)
    cv2.imshow('Fronteira', fronteira)
    cv2.waitKey(0)

    cnts = cv2.findContours(fronteira.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    for con in cnts:
        perimeter = cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, 0.02 * perimeter, True)

        if len(approx) == 4:
            screenCnt = approx
            break
    
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * proporcao)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 15, offset = 10, method = 'gaussian')
    warped = (warped > T).astype('uint8') * 255

    cv2.imshow('Original', img)
    cv2.imshow('Fronteira', fronteira)
    cv2.imshow('Marcado', cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2))
    cv2.imshow('Final', warped)
    cv2.waitKey(0)

def callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        imgcp = im.copy()
        for pt in pts:
            (pX, pY) = pt
            cv2.circle(imgcp, (pX, pY), 10, (0, 255, 0), -1)
            if (len(pts) == 4):
                pts_np = np.array(pts, dtype = 'float32')
                warped = four_point_transform(im.copy(), pts_np)
                warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                # warped = cv2.GaussianBlur(warped, (3, 3), 0)
                cv2.imshow('Warped', imutils.resize(warped, height=600))
        cv2.imshow('Imagem', imgcp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--imagem', type = str, default = 'imagens/imagem_1.png',
                    help = 'caminho da imagem')
    ap.add_argument('-m', '--modo', type = str, default = 'automatico',
                    help = 'modo de seleção do documento')
    args = vars(ap.parse_args())

    if (args['modo'] == 'automatico'):
        img = cv2.imread(args['imagem'])
        selecionar_documento(img.copy())
    else:
        global im, nome_janela, pts

        pts = []
        nome_janela = 'Imagem'
        im = imutils.resize(cv2.imread(args['imagem']), height = 400)
        cv2.imshow(nome_janela, im)
        cv2.setMouseCallback(nome_janela, callback)
        cv2.waitKey(0)

main()

































    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # hist /= hist.sum()
    # hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # plt.imshow(img, cmap='gray')
    # cv2.imshow('Rescaled', img)
    # cv2.waitKey(0)

#     cv2.imshow('Original', img)
#     cv2.waitKey(0)

#     plt.figure()
#     plt.title('Hist Normalizado')
#     plt.xlabel('Bins')
#     plt.ylabel('% of pixels')
#     plt.plot(hist)
#     plt.xlim([0, 256])
#     plt.show()
    
