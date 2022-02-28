import os
import cv2
import pickle
import importlib.util
from extratores.extrator_texto import PytesseractOCR, TorchOCR
from pre_ocr import retirar_sombras


def salvar_palavras():
    resultado = dict()

    for num in range(1, 126):
        caminho_input = 'imagens/Notificacoes/{}.png'.format(num)
        img = cv2.imread(caminho_input)
        img = retirar_sombras(img)

        extrator = TorchOCR()
        palavras = extrator(img)

        resultado[num] = {'palavras': palavras}

    salvar_dados(resultado, 'fixtures/', 'palavras_erode_dilate')


def reestrurar_dados():
    caminho_palavras = 'fixtures/'
    dicionario_palavras = dict()

    for arquivo in os.listdir(caminho_palavras):
        if os.path.isfile(os.path.join(caminho_palavras, arquivo)):
            modulo = importlib.util.spec_from_file_location(
                'lista_palavras',
                '{}{}'.format(caminho_palavras, arquivo))

            lista_palavras = importlib.util.module_from_spec(modulo)
            modulo.loader.exec_module(lista_palavras)
            lista_palavras = lista_palavras.lista_palavras

            dicionario_palavras[int(arquivo.split('.')[0].split('_')[1])] = {'palavras': lista_palavras}
            salvar_dados(dicionario_palavras, 'fixtures/', 'palavras')


def salvar_dados(dados, caminho_saida, nome):
    try:
        with open('{}{}.pickle'.format(caminho_saida, nome), 'wb') as f:
            pickle.dump(dados, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print('Erro ao salvar arquivo com pickle.')


def carregar_dados(caminho):
    try:
        with open(caminho, 'rb') as f:
            return pickle.load(f)
    except Exception as ex:
        print('Erro ao carregar arquivo com pickle.')


# reestrurar_dados()
# salvar_palavras()
