import os
import cv2
import csv
import pickle
import unicodedata
import numpy as np
import pandas as pd
from os.path import exists
from pre_processadores.pre_processador import PreProcessador
from detectores.detector_documento import DetectorLRDECustomizado
from extratores.extrator_documento import ExtratorDocumento
from extratores.extrator_texto import TorchOCR, PytesseractOCR
from extratores.extrator_informacao import IERegrado

'''
Classe responsável por organizar todo o processo de:
    -> Pré-processamento
    -> Detecção de Documento
    -> Extração do Documento
    -> Extração de Texto
    -> Extração das informações do formulário do MultasBR
Retorna uma instância contendo as informações relevantes para o formulário do MultasBR ou, de acordo,
com o método escolhido, salva arquivos relevantes para o desenvolvimento e teste desta classe.
'''

class InformacoesRecurso:
    def __init__(self):
        self.pre_processamento = PreProcessador()  # Operações morfológicas de pre-processamento da imagem
        self.detector_documento = DetectorLRDECustomizado()  # Instância do Detector de documento por Hough
        self.extrator_documento = ExtratorDocumento()  # Instância do extrator do documento da imagem
        self.extrator_texto = TorchOCR()  # Instância do extrator de texto no documento
        self.extrator_informacao = IERegrado()  # Instância de extrator de informações relevantes

    def __call__(self, caminho, teste_mascara=False, teste_ajuste_roi=False):
        if teste_mascara:
            img_original, documento = self.__processamento_comum(caminho, teste_mascara)
            self.__testar_mascaras(caminho, img_original, documento)
            return
        elif teste_ajuste_roi:
            img_original, documento = self.__processamento_comum(caminho)
            cv2.imwrite('imagens/notificacoes_teste/imagens_modificadas/' + caminho.split('/')[-1], documento)
        else:
            img_original, documento = self.__processamento_comum(caminho)
            palavras = self.extrator_texto(documento)
            return palavras

    def __processamento_comum(self, caminho, e_teste=False) -> tuple:
        img_original = cv2.imread(caminho)

        img, ratio = self.pre_processamento(img_original)
        vertices = self.detector_documento(img.copy())
        documento = self.extrator_documento(img, vertices, img_original, ratio, e_teste)

        return img_original, documento

    @staticmethod
    def __testar_mascaras(caminho, img_original, documento) -> None:
        nomes = caminho.split('/')
        nome_arquivo = nomes[3].split('.')
        resultado = np.zeros(img_original.shape[:2])
        quad = np.array(documento, dtype='int32').reshape(4, 2)

        cv2.fillPoly(resultado, pts=[quad], color=(255, 255, 255))
        cv2.imwrite(nomes[0] + '/' + nomes[1] + '/' + nomes[2] + '/Mascaras/' + nome_arquivo[0]
                    + '_M.' + nome_arquivo[1], resultado)

    @staticmethod
    def iou(caminho_mascaras_automatico, medir_validacao=False, medir_teste=False):
        caminho_manual = caminho_mascaras_automatico + 'Mascaras/'
        arquivo_output = 'Resultado_IoU.csv'
        valores_iou = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        resultados = []

        if medir_validacao and medir_teste:
            caminho_completo = ['Treinamento/Mascaras/', 'Validacao/Mascaras/', 'Teste/Mascaras/']
        elif medir_teste:
            caminho_completo = ['Treinamento/Mascaras/', 'Teste/Mascaras/']
        elif medir_validacao:
            caminho_completo = ['Treinamento/Mascaras/', 'Validacao/Mascaras/']
        else:
            caminho_completo = ['Treinamento/Mascaras/']

        for c in caminho_completo:
            for arquivo in os.listdir(caminho_mascaras_automatico + c):
                if os.path.isfile(os.path.join(caminho_mascaras_automatico + c, arquivo)):
                    auto = cv2.imread(caminho_mascaras_automatico + c + arquivo)
                    manual = cv2.imread(caminho_manual + arquivo)

                    intersecao = cv2.bitwise_and(auto, manual)
                    uniao = cv2.bitwise_or(auto, manual)

                    iou = np.count_nonzero(intersecao == 255) / np.count_nonzero(uniao == 255)

                    if c.__contains__('Treinamento'):
                        tipo = 'Treinamento'
                    elif c.__contains__('Validacao'):
                        tipo = 'Validacao'
                    else:
                        tipo = 'Teste'

                    resultados.append([tipo, arquivo, iou])

        campos = ['Tipo', 'Imagem', 'IoU']

        for val in valores_iou:
            for vaL_resultado in resultados:
                if vaL_resultado[2] >= val:
                    vaL_resultado.append(1)
                else:
                    vaL_resultado.append(0)

            campos.append('IoU - ' + str(val))

        with open(arquivo_output, 'w') as f:
            write = csv.writer(f)
            write.writerow(campos)
            write.writerows(resultados)

        return

    def desempenho_ocr(self):
        caminho_arquivo_extrator = 'resultados.csv'
        caminho_desempenho_ocr = 'desempenho_ocr.csv'
        caminho_base_dados = 'Base_de_Dados.csv'
        caminho_indicadores_ocr = 'indicadores_ocr.csv'

        if not exists(caminho_arquivo_extrator) and not exists(caminho_desempenho_ocr):
            palavras_formulario = list()
            desempenho_ocr = list()
            conjunto_palavras = self.carregar_palavras()

            for nome, palavras in zip(conjunto_palavras.keys(), conjunto_palavras.values()):
                img = cv2.imread('imagens/notificacoes_teste/imagens_modificadas/' + nome)
                extrator_informacao = IERegrado()

                extrator_informacao(palavras['palavras'], img)

                resultado = dict()
                resultado_desempenho_ocr = dict()
                formulario = extrator_informacao.informacoes_formulario

                for chv, vlr in zip(formulario.keys(), formulario.values()):
                    resultado[chv] = vlr.texto
                    resultado_desempenho_ocr[chv] = extrator_informacao.informacao_desempenho[chv]
                resultado['arquivo'] = nome
                resultado_desempenho_ocr['arquivo'] = nome

                palavras_formulario.append(resultado)
                desempenho_ocr.append(resultado_desempenho_ocr)

            nome_colunas = ['arquivo']
            nome_colunas.extend(extrator_informacao.informacoes_formulario.keys())
            nome_colunas.insert(4, 'proprietario_pessoa')
            nome_colunas.insert(8, 'condutor_pessoa')
            nome_colunas = nome_colunas[:len(nome_colunas) - 2]

            with open(caminho_arquivo_extrator, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=nome_colunas)
                writer.writeheader()
                writer.writerows(palavras_formulario)

            with open(caminho_desempenho_ocr, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=nome_colunas)
                writer.writeheader()
                writer.writerows(desempenho_ocr)

        if not exists(caminho_indicadores_ocr):
            l_percent_loc, m_porcen_loc = self.indicador_percentual_chave_localizada(
                caminho_desempenho_ocr,
                caminho_base_dados
            )
            l_percent_preen, m_porcen_preen = self.indicador_percentual_chave_preenchida(
                caminho_arquivo_extrator,
                caminho_base_dados
            )
            l_percent_correto, m_porcen_correto = self.indicador_percentual_campo_correto(
                caminho_arquivo_extrator,
                caminho_base_dados
            )

            lista_resultado = list()
            nome_colunas = [
                'arquivo',
                'percentual_chave_localizada',
                'media_chave_localizada',
                'percentual_chave_preenchida',
                'media_chave_preenchida',
                'percentual_preenchimento_correto',
                'media_preenchimento_correto'
            ]
            for val_1, val_2, val_3 in zip(l_percent_loc, l_percent_preen, l_percent_correto):
                resultado = dict()

                resultado[nome_colunas[0]] = val_1[0]
                resultado[nome_colunas[1]] = val_1[1]
                resultado[nome_colunas[2]] = m_porcen_loc
                resultado[nome_colunas[3]] = val_2[1]
                resultado[nome_colunas[4]] = m_porcen_preen
                resultado[nome_colunas[5]] = val_3[1]
                resultado[nome_colunas[6]] = m_porcen_correto

                lista_resultado.append(resultado)

            with open(caminho_indicadores_ocr, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=nome_colunas)
                writer.writeheader()
                writer.writerows(lista_resultado)

    @staticmethod
    def indicador_percentual_chave_localizada(caminho_desempenho_ocr, caminho_base):
        df_manual = pd.read_csv(caminho_base)
        df_programa = pd.read_csv(caminho_desempenho_ocr)

        lista_valores_manual = list()
        lista_valores = list()

        for idx in range(0, df_programa.shape[0]):
            lista_valores_manual.append(df_manual.iloc[idx].tolist())
            lista_valores.append(df_programa.iloc[idx].tolist())

        lista_resultado = list()
        for valores, val_manual in zip(lista_valores, lista_valores_manual):
            lista_resultado.append(
                (
                    valores[0],
                    (sum([1 for val in valores[1:] if val]) /
                    sum([1 for val in val_manual[1:] if not pd.isna(val) and not val == 0])) * 100
                )
            )

        media_porcentagem = sum([val[1] for val in lista_resultado]) / len(lista_resultado)

        return lista_resultado, media_porcentagem

    def indicador_percentual_chave_preenchida(self, caminho_extraido, caminho_base):
        df_manual = pd.read_csv(caminho_base)
        df_programa = pd.read_csv(caminho_extraido)

        valor_manual = self.__nome_arquivo_numero_valores(df_manual)
        valor_programa = self.__nome_arquivo_numero_valores(df_programa)

        lista_resultado = list()
        for t_1, t_2 in zip(valor_manual, valor_programa):
            lista_resultado.append((t_1[0], (t_2[1] / t_1[1]) * 100))

        media_porcentagem = sum([val[1] for val in lista_resultado]) / len(lista_resultado)

        return lista_resultado, media_porcentagem

    def indicador_percentual_campo_correto(self, caminho_extraido, caminho_base):
        df_manual = pd.read_csv(caminho_base)
        df_programa = pd.read_csv(caminho_extraido)

        resultado = list()
        for idx in range(0, df_manual.shape[0]):
            iguais = 0
            l_man = [val for val in df_manual.iloc[idx]]
            l_pro = [val for val in df_programa.iloc[idx]]

            for v_m, v_p in zip(l_man[1:], l_pro[1:]):
                if pd.isna(v_m) and pd.isna(v_p):
                    iguais += 0
                elif pd.isna(v_m) and not pd.isna(v_p):
                    iguais += 0
                elif not pd.isna(v_m) and pd.isna(v_p):
                    iguais += 0
                else:
                    if self.__e_igual(str(v_m), str(v_p)):
                        iguais += 1

            resultado.append((int(l_man[0]), (iguais / len(l_man[1:]) * 100)))

        media_porcentagem = sum([val[1] for val in resultado]) / len(resultado)

        return resultado, media_porcentagem

    @staticmethod
    def __e_igual(p1, p2):
        return unicodedata \
            .normalize('NFD', p1) \
            .encode('ascii', 'ignore') \
            .decode('utf-8') \
            .lower() \
            .strip() \
            .__contains__(unicodedata
                          .normalize('NFD', p2)
                          .encode('ascii', 'ignore')
                          .decode('utf-8')
                          .lower()
                          .strip())

    def indicador_chaves_encontradas(self, caminho_extraido, caminho_base):
        df_manual = pd.read_csv(caminho_base)
        df_programa = pd.read_csv(caminho_extraido)

        valor_manual = self.__nome_arquivo_numero_valores(df_manual)
        valor_programa = self.__nome_arquivo_numero_valores(df_programa)

        lista_resultado = list()
        for t_1, t_2 in zip(valor_manual, valor_programa):
            lista_resultado.append((t_1[0], t_2[1] * 100 / t_1[1]))

        media_porcentagem = sum([val[1] for val in lista_resultado]) / len(lista_resultado)

        return lista_resultado, media_porcentagem

    @staticmethod
    def __nome_arquivo_numero_valores(df):
        resultado = list()
        for idx in range(0, df.shape[0]):
            lista_valores = df.iloc[idx].tolist()
            valores = len(list(filter(lambda val: not pd.isna(val), lista_valores))) - 1

            resultado.append((lista_valores[0], valores))

        return resultado

    @staticmethod
    def salvar_palavras(resultado, caminho_saida, nome):
        try:
            with open('{}{}.pickle'.format(caminho_saida, nome), 'wb') as f:
                pickle.dump(resultado, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print('Erro ao salvar arquivo com pickle.')

    @staticmethod
    def carregar_palavras(caminho='fixtures/palavras_teste_final.pickle'):
        try:
            with open(caminho, 'rb') as f:
                return pickle.load(f)
        except Exception as ex:
            print('Erro ao carregar arquivo com pickle.')

    def marcar_chaves(self):
        conjunto_palavras = self.carregar_palavras()

        for nome, palavras in zip(conjunto_palavras.keys(), conjunto_palavras.values()):
            img = cv2.imread('imagens/notificacoes_teste/imagens_modificadas/' + nome)
            extrator_informacao = IERegrado()

            extrator_informacao(palavras['palavras'], img)
            extrator_informacao.marcar_palavras()
            cv2.imwrite('imagens/notificacoes_teste/imagens_marcadas/' + nome, extrator_informacao.img)