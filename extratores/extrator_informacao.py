import re
import cv2
import copy
import unicodedata
from entidades.palavra import Palavra


class IERegrado:
    def __init__(self):
        self.e_pesquisa_vertical = True
        self.estado_re = re.compile(r"[A-Z]{2}\b")
        self.cpf_re = re.compile(r"\d{3}[.]?\d{3}[.]?\d{3}[-]?\d{2}")
        self.cnpj_re = re.compile(r"\d{2}[.]?\d{3}[.]?\d{3}[/]?\d{4}[-]?\d{2}")
        self.placa_re = re.compile(r"\D{3}[-]?\d{4}|\D{3}\d\D\d{2}")
        self.gravidades_re = re.compile(r"leve|m[eé]dia|grave|grav[ií]ssima", flags=re.I)
        self.estados_brasileiros_re = re.compile(
            r"RR|AP|AM|PA|AC|RO|TO|MA|PI|CE|RN|PB|PE|AL|SE|BA|MT|DF|GO|MS|MG|ES|RJ|SP|PR|SC|RS")
        self.informacoes_relevantes = {
            'proprietario': {'nome': None, 'cpf': None, 'cnpj': None},
            'condutor': {'nome': None, 'cpf': None, 'cnpj': None},
            # 'veiculo': {'placa': None, 'marca': None, 'renavam': None},
            'placa': None,
            'marca': None,
            'modelo': None,
            'renavam': None,
            'local': {'uf': None},
            'uf': None,
            'gravidade': None,
            'natureza': None,
        }

    def __call__(self, palavras, img):
        self.palavras = palavras
        self.img = img

        resultado_vertical = self.atravessar_dicio(
            copy.deepcopy(self.informacoes_relevantes),
            self.pesquisa,
            self.pesquisa
        )
        self.e_pesquisa_vertical = False
        resultado_horizontal = self.atravessar_dicio(
            copy.deepcopy(self.informacoes_relevantes),
            self.pesquisa,
            self.pesquisa
        )

        self.informacoes_relevantes = self.comparar_valores_capturados(
            copy.deepcopy(self.informacoes_relevantes),
            resultado_vertical,
            resultado_horizontal
        )
        self.informacoes_relevantes = self.comparar_proprietario_condutor(
            copy.deepcopy(self.informacoes_relevantes)
        )
        self.informacoes_relevantes = self.determinar_pessoa(
            copy.deepcopy(self.informacoes_relevantes)
        )

    def coletar_resultados(self, locais, campo):
        frases_selecionadas = list()

        for campo_dict in locais.values():
            if self.e_pesquisa_vertical:
                palavras_proxima_linha = list(filter(lambda palavra:
                                                     palavra.linha == campo_dict.linha + 1
                                                     and palavra.bloco == campo_dict.bloco,
                                                     self.palavras))
            else:
                palavras_proxima_linha = list(filter(lambda palavra:
                                                     palavra.linha == campo_dict.linha
                                                     and palavra.bloco == campo_dict.bloco
                                                     and palavra.localizacao[0] != campo_dict.localizacao[0],
                                                     self.palavras))

            palavras_proxima_linha = self.verificar_palavras(campo, palavras_proxima_linha)

            if len(palavras_proxima_linha) == 0:
                continue

            frases = self.separar_em_frases(palavras_proxima_linha)

            temp = []
            for idx, frase in enumerate(frases):
                temp.append((idx, abs((frase[0].localizacao[0][0] / campo_dict.localizacao[0][0]) - 1)))

            temp.sort(key=lambda val: val[1])
            frases_selecionadas.append(frases[temp[0][0]])

        return frases_selecionadas

    def comparar_valores_capturados(self, resultado, r_vertical, r_horizontal):
        for chave in resultado:
            if type(resultado[chave]) is dict:
                for chave_interna in resultado[chave]:
                    v_1, v_2 = r_vertical[chave][chave_interna], r_horizontal[chave][chave_interna]
                    resultado[chave][chave_interna] = self.testar_valores(v_1, v_2)
            else:
                v_1, v_2 = r_vertical[chave], r_horizontal[chave]
                resultado[chave] = self.testar_valores(v_1, v_2)

        return resultado

    def extrair_palavras(self, campo, texto):
        if campo == 'cpf':
            return self.cpf_re.search(texto).group()
        elif campo == 'cnpj':
            return self.cnpj_re.search(texto).group()
        elif campo == 'placa':
            return self.placa_re.search(texto).group()
        elif campo == 'uf':
            estados = self.estados_brasileiros_re.findall(texto)

            if len(estados) == 0:
                return []
            elif len(estados) > 1:
                return estados[-1]
            else:
                return estados[0]
        elif campo == 'gravidade' or campo == 'natureza':
            return self.gravidades_re.search(texto).group()
        else:
            return texto

    def filtrar_palavras_encontradas(self, word):
        resultado = list(filter(lambda palavra:
                                unicodedata
                                .normalize('NFD', palavra.texto)
                                .encode('ascii', 'ignore')
                                .decode('utf-8')
                                .lower()
                                .strip()
                                .__contains__(word),
                                self.palavras))

        resultado.sort(key=lambda palavra: palavra.localizacao[0][0])
        return resultado

    def pesquisa(self, dicio, assunto, campo=None):
        locais_assunto = self.filtrar_palavras_encontradas(assunto)

        if campo is not None:
            locais_campo = self.filtrar_palavras_encontradas(campo)

            if len(locais_campo) != 0:
                locais_palavras_interesse = self.determinar_conjunto_proximo(locais_assunto, locais_campo)
                dicio[assunto][campo] = self.preenche_informacao(locais_palavras_interesse, assunto, campo)

            return

        if len(locais_assunto) != 0:
            locais_palavras_interesse = {i: locais_assunto[i] for i in range(0, len(locais_assunto))}
            dicio[assunto] = self.preenche_informacao(locais_palavras_interesse, assunto)

    def preenche_informacao(self, locais_interesse, assunto, campo=None):
        if campo is None:
            frases_selecionadas = self.coletar_resultados(locais_interesse, assunto)
        else:
            frases_selecionadas = self.coletar_resultados(locais_interesse, campo)

        resultados = list()
        if len(frases_selecionadas) != 0:
            for frase in frases_selecionadas:
                resultado = ''
                for palavra in frase:
                    resultado += palavra.texto + ' '

                resultado = resultado.strip()

                if campo is None:
                    resultado = self.extrair_palavras(assunto, resultado)
                else:
                    resultado = self.extrair_palavras(campo, resultado)

                if len(resultado) > 0:
                    resultados.append(resultado)

        if len(resultados) != 0:
            return Palavra(
                resultados[0],
                (frases_selecionadas[0][0].localizacao[0], frases_selecionadas[0][-1].localizacao[1]),
                frases_selecionadas[0][0].linha,
                frases_selecionadas[0][0].linha_localizacao,
                frases_selecionadas[0][0].bloco,
            )
        else:
            return None

    def verificar_palavras(self, campo, palavras):
        if campo == 'cpf':
            return list(filter(lambda val: self.cpf_re.search(val.texto), palavras))
        elif campo == 'cnpj':
            return list(filter(lambda val: self.cnpj_re.search(val.texto), palavras))
        elif campo == 'placa':
            return list(filter(lambda val: self.placa_re.search(val.texto), palavras))
        elif campo == 'uf':
            return list(filter(lambda val: self.estado_re.search(val.texto), palavras))
        elif campo == 'gravidade' or campo == 'natureza':
            return list(filter(lambda val: self.gravidades_re.search(val.texto), palavras))
        else:
            return palavras

    def expandir_informacoes(self):
        return {
            'p_nome': self.informacoes_relevantes['proprietario']['nome'],
            'p_cpf': self.informacoes_relevantes['proprietario']['cpf'],
            'p_cnpj': self.informacoes_relevantes['proprietario']['cnpj'],
            'p_pessoa': self.informacoes_relevantes['proprietario']['pessoa'],
            'c_nome': self.informacoes_relevantes['condutor']['nome'],
            'c_cpf': self.informacoes_relevantes['condutor']['cpf'],
            'c_cnpj': self.informacoes_relevantes['condutor']['cnpj'],
            'c_pessoa': self.informacoes_relevantes['condutor']['pessoa'],
            'v_placa': self.informacoes_relevantes['placa'],
            'v_marca': self.informacoes_relevantes['marca'],
            'v_modelo': self.informacoes_relevantes['modelo'],
            'v_renavam': self.informacoes_relevantes['renavam'],
            'l_uf': self.informacoes_relevantes['local']['uf'],
            'uf': self.informacoes_relevantes['uf'],
            'gravidade': self.informacoes_relevantes['gravidade'],
            'natureza': self.informacoes_relevantes['natureza'],
        }

    def marcar_palavras(self):
        dicio = self.expandir_informacoes()
        for chave, valor in zip(dicio.keys(), dicio.values()):
            if valor is None or type(valor) == str:
                continue

            cv2.rectangle(
                self.img,
                valor.localizacao[0],
                valor.localizacao[1],
                (255, 0, 0),
                2,
            )
            cv2.putText(
                self.img,
                chave,
                valor.localizacao[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

    @staticmethod
    def atravessar_dicio(dicio, func_1, func_2):
        for chv_ext in dicio:
            if type(dicio[chv_ext]) is dict:
                for chv_int in dicio[chv_ext]:
                    func_1(dicio, chv_ext, chv_int)
            else:
                func_2(dicio, chv_ext)

        return dicio

    @staticmethod
    def comparar_proprietario_condutor(dicio):
        dict_proprietario = dicio['proprietario']
        dict_condutor = dicio['condutor']

        for chv in dict_proprietario:
            if dict_proprietario[chv] is not None and dict_condutor[chv] is not None:
                if dict_proprietario[chv].texto == dict_condutor[chv].texto:
                    dicio['condutor'] = dict_proprietario
                    break
                elif len(dict_condutor[chv].texto) == 0:
                    dicio['condutor'][chv] = dict_proprietario[chv]
                elif len(dict_proprietario[chv].texto) == 0:
                    dicio['proprietario'][chv] = dict_condutor[chv]

        return dicio

    @staticmethod
    def determinar_conjunto_proximo(locais_assunto, locais_campo):
        locais = {}

        for assunto in locais_assunto:
            temp = []

            for idx, campo in enumerate(locais_campo):
                temp.append((
                    idx,
                    campo.localizacao[0][1] - assunto.localizacao[0][1]
                ))

            temp = list(filter(lambda val: val[1] > 0, temp))
            if len(temp) == 0:
                continue

            temp.sort(key=lambda val: val[1])
            locais[assunto.localizacao[0][1]] = locais_campo[temp[0][0]]

        return locais

    @staticmethod
    def determinar_pessoa(dicio):
        proprietario_cpf = dicio['proprietario']['cpf'].texto if dicio['proprietario']['cpf'] is not None else ''
        condutor_cpf = dicio['condutor']['cpf'].texto if dicio['condutor']['cpf'] is not None else ''

        proprietario_cnpj = dicio['proprietario']['cnpj'].texto if dicio['proprietario']['cnpj'] is not None else ''
        condutor_cnpj = dicio['condutor']['cnpj'].texto if dicio['condutor']['cnpj'] is not None else ''

        if len(proprietario_cpf) > 0:
            dicio['proprietario']['pessoa'] = 'Física'
        elif len(proprietario_cnpj) > 0:
            dicio['proprietario']['pessoa'] = 'Jurídica'
        else:
            dicio['proprietario']['pessoa'] = ''

        if condutor_cpf is not None and len(condutor_cpf) > 0:
            dicio['condutor']['pessoa'] = 'Física'
        elif len(condutor_cnpj) > 0:
            dicio['condutor']['pessoa'] = 'Jurídica'
        else:
            dicio['condutor']['pessoa'] = ''

        return dicio

    @staticmethod
    def separar_em_frases(palavras):
        espaco = 20
        palavras = palavras.copy()
        palavras.sort(key=lambda p: p.localizacao[0][0])

        resultado = []
        frase = list()
        for idx, palavra in enumerate(palavras):
            if len(frase) == 0:
                frase.append(palavra)

            if idx + 1 == len(palavras):
                resultado.append(frase.copy())
                frase.clear()
                break

            if palavras[idx + 1].localizacao[0][0] - palavra.localizacao[1][0] <= espaco:
                frase.append(palavras[idx + 1])
            else:
                resultado.append(frase.copy())
                frase.clear()

        return resultado

    @staticmethod
    def testar_valores(valor_1, valor_2, campo=None):
        resultado = None

        if valor_1 is None:
            resultado = valor_2
        elif valor_2 is None:
            resultado = valor_1
        elif valor_1.texto == valor_2.texto:
            resultado = valor_1
        else:
            resultado = valor_1

        return resultado
