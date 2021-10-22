import cv2
import imutils
import argparse
import extrator_info as ext

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--imagem', type = str, default = 'imagens/folha_1.jpeg',
                help = 'caminho da imagem')
ap.add_argument('-m', '--modo', type = str, default = 'auto',
                help = 'modo detecção das fronteiras')
args = vars(ap.parse_args())

pontos = None
nova_altura = 800
nome_janela = 'Selecao'
org_img = cv2.imread(args['imagem']) 
ratio = org_img.shape[0] / float(nova_altura)
img = imutils.resize(org_img, height = nova_altura, inter = cv2.INTER_CUBIC)

if args['modo'] == 'auto':
    pontos = ext.ExtratorInfo().detectar_automaticamente(img)

cv2.imshow(nome_janela, img)
roi = ext.RoiAjustavel(img, nome_janela, img.shape[1], img.shape[0], pontos)
cv2.setMouseCallback(nome_janela, ext.arrastarQuad, roi)
cv2.namedWindow(nome_janela)
roi.selecionarROI()
roi.imagemProcessada(org_img, ratio)
