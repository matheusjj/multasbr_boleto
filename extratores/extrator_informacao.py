import cv2
import pytesseract
from pytesseract import Output
from entidades.informacao_relevante import InformacaoRelevante


class DetectorInformacao:
    def __init__(self, psm='4'):
        self.psm = psm
        self.informacoes_relevantes = {
            'proprietário': {'nome': None, 'cpf': None, 'cnpj': None, 'pessoa': None},
            'condutor': {'nome': None, 'cpf': None, 'cnpj': None, 'pessoa': None},
            'uf': None,
            'categoria': None,
            'natureza': None,
        }

    def __call__(self, imagem):
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        options = '-l {} --psm {}'.format('por', self.psm)
        palavras_capturadas = pytesseract.image_to_data(imagem,
                                                        output_type=Output.DICT, config=options)

        palavras = {
            'text': [],
            'line_num': [],
            'word_num': [],
            'top': [],
            'left': [],
            'width': [],
        }

        for idx in range(0, len(palavras_capturadas['text'])):
            if len(palavras_capturadas['text'][idx].strip()) != 0:
                palavras['text'].append(palavras_capturadas['text'][idx])
                palavras['line_num'].append(palavras_capturadas['line_num'][idx])
                palavras['word_num'].append(palavras_capturadas['word_num'][idx])
                palavras['top'].append(palavras_capturadas['top'][idx])
                palavras['left'].append(palavras_capturadas['left'][idx])
                palavras['width'].append(palavras_capturadas['width'][idx])

        self.__informacoes_proprietario(palavras)
        self.__informacoes_condutor(palavras)

    def __buscar_informacao_abaixo(self, palavras_encontradas, assunto, info):
        locais_assunto = []
        locais_info = []

        for idx in range(0, len(palavras_encontradas['text'])):
            texto, num_linha, num_palavra, topo, esquerda, largura = \
                palavras_encontradas['text'][idx], palavras_encontradas['line_num'][idx], \
                palavras_encontradas['word_num'][idx], palavras_encontradas['top'][idx], \
                palavras_encontradas['left'][idx], palavras_encontradas['width'][idx]

            if texto.lower().strip().__contains__(assunto):
                locais_assunto.append(InformacaoRelevante(texto, num_linha, num_palavra, topo, esquerda, largura))
            elif texto.lower().strip().__contains__(info):
                locais_info.append(InformacaoRelevante(texto, num_linha, num_palavra, topo, esquerda, largura))

        # TODO: E se uma ou ambas as listas forem vazias
        locais_assunto.sort(key=lambda val: val.topo)
        locais_info.sort(key=lambda val: val.topo)

        dict_locais = {}

        for val_assunto in locais_assunto:
            temp = []

            for (idx, val_info) in enumerate(locais_info):
                temp.append((idx, val_info.topo - val_assunto.topo))

            # TODO: E se todos os valores forem negativos
            temp = list(filter(lambda val: val[1] > 0, temp))
            if len(temp) == 0:
                continue

            temp.sort(key=lambda val: val[1])
            dict_locais[val_assunto.topo] = temp[0]

        minima_distancia = min([val[1] for val in list(dict_locais.values())])
        chaves_valor_minimo = [chave for chave in dict_locais if dict_locais[chave][1] == minima_distancia]
        posicao_relevante = (chaves_valor_minimo[0], locais_info[dict_locais[chaves_valor_minimo[0]][0]])

        linhas_possiveis = list(filter(lambda val: val > posicao_relevante[1].topo,
                                       list(dict.fromkeys(palavras_encontradas['top']))))[:5]

        palavras_selecionadas = []
        for idx in range(0, len(palavras_encontradas['text'])):
            if palavras_encontradas['top'][idx] in linhas_possiveis:
                informacao = InformacaoRelevante(
                    palavras_encontradas['text'][idx], palavras_encontradas['line_num'][idx],
                    palavras_encontradas['word_num'][idx], palavras_encontradas['top'][idx],
                    palavras_encontradas['left'][idx], palavras_encontradas['width'][idx])

                palavras_selecionadas.append(informacao)

        palavras_selecionadas = InformacaoRelevante.retirar_vazios(palavras_selecionadas)
        palavras_selecionadas = InformacaoRelevante.selecionar_mesma_linha(palavras_selecionadas)
        resultado = InformacaoRelevante.determinar_frases(palavras_selecionadas, posicao_relevante[1].esquerda)
        print('oi')



    def __informacoes_proprietario(self, palavras_encontradas):
        self.__buscar_informacao_abaixo(palavras_encontradas, 'infração', 'uf')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'infração', 'categoria')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'infração', 'natureza')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'condutor', 'pessoa')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'condutor', 'nome')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'condutor', 'cpf')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'proprietário', 'pessoa')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'proprietário', 'nome')
        self.__buscar_informacao_abaixo(palavras_encontradas, 'proprietário', 'cpf')

        # limite_superior = None
        # limite_inferior = None

        # chaves = list(palavras_encontradas.keys())
        # for idx in range(0, len(chaves)):
        #     print(chaves[idx], palavras_encontradas[chaves[idx]])
        # for idx in range(0, len(palavras_encontradas['text'])):
        #     if palavras_encontradas['text'][idx].lower() == 'dados' or palavras_encontradas['text'][idx].lower() == 'do' or palavras_encontradas['text'][idx].lower() == 'proprietário':
        #         print(palavras_encontradas['text'][idx], palavras_encontradas['line_num'][idx], palavras_encontradas['top'][idx])

        # for idx in range(0, len(palavras_encontradas['text'])):
        #     if palavras_encontradas['text'][idx].lower() == 'proprietário':
        #         limite_superior = (idx, palavras_encontradas['top'][idx])
        #     elif palavras_encontradas['text'][idx].lower() == 'condutor':
        #         limite_inferior = (idx, palavras_encontradas['top'][idx])
        #     elif limite_superior is not None and limite_inferior is not None:
        #         break
        #
        # print(limite_superior, limite_inferior)
        #
        # for idx in range(0, len(palavras_encontradas['text'])):
        #     pass
        #

    def __informacoes_condutor(self, palavras_encontradas):
        pass
