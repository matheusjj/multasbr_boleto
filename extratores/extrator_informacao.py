import re
import cv2
import copy
import unicodedata
from entidades.palavra import Palavra

'''
Classe responsável por filtrar as informações relevantes para o preenchimento do formulário do MultasBR
Retorna self
'''

class IERegrado:
    def __init__(self):
        self.e_pesquisa_vertical = True
        self.estado_re = re.compile(r"[A-Z]{2}\b")
        self.cpf_re = re.compile(r"^\d{3}[.]\d{3}[.]\d{3}[-]\d{2}$")
        self.cnpj_re = re.compile(r"^\d{2}[.]?\d{3}[.]?\d{3}[/]?\d{4}[-]?\d{2}$")
        self.placa_re = re.compile(r"^[A-Za-z]{3}[-]?\d{4}$|^[A-Za-z]{3}\d[A-Za-z]\d{2}$")
        self.gravidades_re = re.compile(r"leve|m[eé]dia|grave|grav[ií]ssima|3|4|5|7", flags=re.I)
        self.estados_brasileiros_re = re.compile(
            r"RR|AP|AM|PA|AC|RO|TO|MA|PI|CE|RN|PB|PE|AL|SE|BA|MT|DF|GO|MS|MG|ES|RJ|SP|PR|SC|RS")
        self.palavras_chave = {
            'proprietario_nome': ('proprietario', ['nome']),
            'proprietario_cpf': ('proprietario', ['cpf']),
            'proprietario_cnpj': ('proprietario', ['cnpj']),
            'condutor_nome': ('condutor', ['nome']),
            'condutor_cpf': ('condutor', ['cpf']),
            'condutor_cnpj': ('condutor', ['cnpj']),
            'veiculo_placa': ['placa'],
            'veiculo_marca': ['marca', 'modelo'],
            'veiculo_renavam': ['renavam'],
            'local_uf': ('local', ['uf', 'rua', 'endereco']),
            'gravidade': ['gravidade', 'categoria', 'natureza', 'pontuacao']
        }
        self.informacoes_formulario = {chv: list() for chv in self.palavras_chave.keys()}
        self.informacao_desempenho = {chv: False for chv in self.palavras_chave.keys()}

    def __call__(self, palavras, img):
        self.palavras = palavras
        self.img = img

        resultado_vertical = {chv: list() for chv in self.palavras_chave.keys()}
        resultado_horizontal = {chv: list() for chv in self.palavras_chave.keys()}

        for dicio in [resultado_vertical, resultado_horizontal]:
            for chv_dicio, palavra_chave in zip(self.palavras_chave.keys(), self.palavras_chave.values()):
                t = self.pesquisar(chv_dicio, palavra_chave)
                dicio[chv_dicio] = list(filter(lambda item: item is not None, t))
            self.e_pesquisa_vertical = not self.e_pesquisa_vertical

        self.informacoes_formulario = self.comparar_valores_capturados(resultado_vertical, resultado_horizontal)
        self.informacoes_formulario = self.comparar_proprietario_condutor(copy.deepcopy(self.informacoes_formulario))
        self.informacoes_formulario = self.corrigir_gravidade(copy.deepcopy(self.informacoes_formulario))
        self.informacoes_formulario = self.corrigir_marca(copy.deepcopy(self.informacoes_formulario))
        self.informacoes_formulario = self.preenchimento_final(copy.deepcopy(self.informacoes_formulario))
        self.informacoes_formulario = self.determinar_pessoa(copy.deepcopy(self.informacoes_formulario))

        if self.informacao_desempenho['proprietario_cpf'] or self.informacao_desempenho['proprietario_cnpj']:
            self.informacao_desempenho['proprietario_pessoa'] = True
        else:
            self.informacao_desempenho['proprietario_pessoa'] = False

        if self.informacao_desempenho['condutor_cpf'] or self.informacao_desempenho['condutor_cnpj']:
            self.informacao_desempenho['condutor_pessoa'] = True
        else:
            self.informacao_desempenho['condutor_pessoa'] = False

    def pesquisar(self, chv_dicio, chv):
        resultado = list()
        chave_auxiliar = None
        chaves = None

        if type(chv) is tuple:
            chave_auxiliar = chv[0]
            chaves = chv[1]
        else:
            chaves = chv

        if chave_auxiliar is not None:
            palavras_auxiliares = self.filtrar_palavras_encontradas(chave_auxiliar)

            for chv in chaves:
                palavras_campo = self.filtrar_palavras_encontradas(chv)

                if len(palavras_campo) != 0:
                    self.informacao_desempenho[chv_dicio] = True
                    locais_interesse = self.determinar_conjunto_proximo(palavras_auxiliares, palavras_campo)
                    resultado.append(self.preenche_informacao(locais_interesse, chv_dicio))

            return resultado

        for chv in chaves:
            palavras_campo = self.filtrar_palavras_encontradas(chv)

            if len(palavras_campo) != 0:
                locais_interesse = {i: palavras_campo[i] for i in range(0, len(palavras_campo))}
                resultado.append(self.preenche_informacao(locais_interesse, chv_dicio))

        return resultado

    def coletar_resultados(self, locais, campo):
        frases_selecionadas = list()

        for palavra_encontrada in locais.values():
            if self.e_pesquisa_vertical:
                # Caso seja pesquisa vertical então são selecionadas as palavras que aparecem na
                # linha subsequente ao campo sendo testado.
                dados_extraidos = list(filter(lambda palavra:
                                              palavra.linha == palavra_encontrada.linha + 1
                                              and palavra.bloco == palavra_encontrada.bloco,
                                              self.palavras))
            else:
                # Análogo para a pesquisa horizontal.
                dados_extraidos = list(filter(lambda palavra:
                                              palavra.linha == palavra_encontrada.linha
                                              and palavra.bloco == palavra_encontrada.bloco
                                              and palavra.localizacao[0] != palavra_encontrada.localizacao[0],
                                              self.palavras))

            dados_extraidos = self.verificar_palavras(campo, dados_extraidos)

            if len(dados_extraidos) == 0:
                continue

            frases = self.separar_em_frases(dados_extraidos)

            temp = []
            for idx, frase in enumerate(frases):
                temp.append((idx, abs((frase[0].localizacao[0][0] / palavra_encontrada.localizacao[0][0]) - 1)))

            temp.sort(key=lambda val: val[1])
            frases_selecionadas.append(frases[temp[0][0]])

        return frases_selecionadas

    def comparar_valores_capturados(self, r_vertical, r_horizontal):
        resultado = {chv: None for chv in r_vertical.keys()}

        for chv in resultado:
            valor_1, valor_2 = r_vertical[chv], r_horizontal[chv]
            resultado[chv] = self.testar_valores(valor_1, valor_2)

        return resultado

    def extrair_palavras(self, campo, texto):
        if campo.__contains__('cpf'):
            return self.cpf_re.search(texto).group()
        elif campo.__contains__('cnpj'):
            return self.cnpj_re.search(texto).group()
        elif campo.__contains__('placa'):
            return self.placa_re.search(texto).group()
        elif campo.__contains__('uf'):
            estados = self.estados_brasileiros_re.findall(texto)

            if len(estados) == 0:
                return []
            elif len(estados) > 1:
                return estados[-1]
            else:
                return estados[0]
        elif campo.__contains__('gravidade'):
            return self.gravidades_re.search(texto).group()
        else:
            return texto

    # Filtra, do conjunto de palavras retornadas pelo OCR, aquela que se adequa ao argumento.
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

    def preenche_informacao(self, locais_interesse, campo):
        frases_selecionadas = self.coletar_resultados(locais_interesse, campo)

        resultados = list()
        if len(frases_selecionadas) != 0:
            for frase in frases_selecionadas:
                resultado = ''
                for palavra in frase:
                    resultado += palavra.texto + ' '

                resultado = resultado.strip()
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
        if campo.__contains__('cpf'):
            return list(filter(lambda val: self.cpf_re.search(val.texto), palavras))
        elif campo.__contains__('cnpj'):
            return list(filter(lambda val: self.cnpj_re.search(val.texto), palavras))
        elif campo.__contains__('placa'):
            return list(filter(lambda val: self.placa_re.search(val.texto), palavras))
        elif campo.__contains__('uf'):
            return list(filter(lambda val: self.estado_re.search(val.texto), palavras))
        elif campo.__contains__('gravidade'):
            return list(filter(lambda val: self.gravidades_re.search(val.texto), palavras))
        else:
            return palavras

    def marcar_palavras(self):
        cores = [
            (153, 0, 0),
            (153, 76, 0),
            (153, 153, 0),
            (76, 153, 0),
            (0, 153, 0),
            (0, 153, 76),
            (0, 153, 153),
            (0, 76, 153),
            (0, 0, 153),
            (76, 0, 153),
            (153, 0, 153),
            (153, 0, 76),
            (0, 0, 0),
        ]

        chvs = list(self.informacoes_formulario.keys())
        for idx in range(0, len(chvs)):
            chv = chvs[idx]
            valor = self.informacoes_formulario[chv]
            cor = cores[idx]

            if len(valor.texto) == 0:
                continue

            ja_preenchido = False
            for alt_chv in chvs[0:idx]:
                if self.informacoes_formulario[alt_chv].localizacao == valor.localizacao:
                    ja_preenchido = True
                    break

            if ja_preenchido:
                cv2.putText(
                    self.img,
                    '*',
                    (valor.localizacao[0][0] - 20, valor.localizacao[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    cor,
                    2,
                    cv2.LINE_AA
                )
            else:
                cv2.rectangle(
                        self.img,
                        valor.localizacao[0],
                        valor.localizacao[1],
                        cor,
                        2,
                    )
                cv2.putText(
                    self.img,
                    chv,
                    valor.localizacao[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    cor,
                    2,
                    cv2.LINE_AA
                )

    @staticmethod
    def comparar_proprietario_condutor(dicio):
        campos_preenchidos_proprietario = len(list(filter(lambda item: len(item) > 0,
                                                          [dicio['proprietario_nome'].texto,
                                                           dicio['proprietario_cpf'].texto,
                                                           dicio['proprietario_cnpj'].texto])))

        campos_preenchidos_condutor = len(list(filter(lambda item: len(item) > 0,
                                                      [dicio['condutor_nome'].texto,
                                                       dicio['condutor_cpf'].texto,
                                                       dicio['condutor_cnpj'].texto])))

        if campos_preenchidos_proprietario == campos_preenchidos_condutor:
            if dicio['proprietario_nome'].texto == dicio['condutor_nome'].texto\
                    or dicio['proprietario_cpf'].texto == dicio['condutor_cpf'].texto\
                    or dicio['proprietario_cnpj'].texto == dicio['condutor_cnpj'].texto:
                dicio['condutor_nome'] = dicio['proprietario_nome']
                dicio['condutor_cpf'] = dicio['proprietario_cpf']
                dicio['condutor_cnpj'] = dicio['proprietario_cnpj']
        elif campos_preenchidos_proprietario > campos_preenchidos_condutor:
            dicio['condutor_nome'] = dicio['proprietario_nome']
            dicio['condutor_cpf'] = dicio['proprietario_cpf']
            dicio['condutor_cnpj'] = dicio['proprietario_cnpj']
        elif campos_preenchidos_proprietario < campos_preenchidos_condutor:
            dicio['proprietario_nome'] = dicio['condutor_nome']
            dicio['proprietario_cpf'] = dicio['condutor_cpf']
            dicio['proprietario_cnpj'] = dicio['condutor_cnpj']

        return dicio

    # Organiza os pares de assunto-campo de acordo com a proximidade entre elas.
    # Retornar <dict<int, Palavra>>, sendo o int o local na vertical da palavra auxiliar.
    def determinar_conjunto_proximo(self, palavras_auxiliares, palavras_campo):
        locais = dict()

        for aux in palavras_auxiliares:
            palavras_proximas = list()

            for idx, campo in enumerate(palavras_campo):
                if self.e_pesquisa_vertical:
                    # Se pesquisa for na vertical, então a ordenação é feita de acordo com a distância
                    # vertical entre o assunto e o campo.
                    palavras_proximas.append((
                        idx,
                        campo.localizacao[0][1] - aux.localizacao[0][1]
                    ))
                else:
                    # Analogamente se a pesquisa for feita na horizontal.
                    palavras_proximas.append((
                        idx,
                        campo.localizacao[0][0] - aux.localizacao[0][0]
                    ))

            palavras_proximas = list(filter(lambda val: val[1] >= 0, palavras_proximas))
            if len(palavras_proximas) == 0:
                continue

            palavras_proximas.sort(key=lambda val: val[1])
            locais[aux.localizacao[0][1]] = palavras_campo[palavras_proximas[0][0]]

        return locais

    def preenchimento_final(self, dicio):
        campos_interesse = ['cpf', 'cnpj', 'placa', 'renavam']

        for chv in dicio:
            for cmp in campos_interesse:
                if chv.__contains__(cmp) and len(dicio[chv].texto) == 0:
                    palavras_selecionadas_re = list()

                    if cmp == campos_interesse[0]:
                        palavras_selecionadas_re = list(filter(lambda val: self.cpf_re.search(val.texto),
                                                               self.palavras))
                    elif cmp == campos_interesse[1]:
                        palavras_selecionadas_re = list(filter(lambda val: self.cnpj_re.search(val.texto),
                                                               self.palavras))
                    elif cmp == campos_interesse[2]:
                        palavras_selecionadas_re = list(filter(lambda val: self.placa_re.search(val.texto),
                                                               self.palavras))

                    if len(palavras_selecionadas_re) != 0:
                        dicio[chv] = palavras_selecionadas_re[0]

        return dicio

    def corrigir_marca(self, dicio):
        marca = dicio['veiculo_marca']
        marca_texto = marca.texto

        resultado = self.placa_re.split(marca_texto)

        if len(resultado) == 1:
            dicio['veiculo_marca'] = Palavra(resultado[0], marca.localizacao, marca.linha,
                                             marca.linha_localizacao, marca.bloco)
        else:
            dicio['veiculo_marca'] = Palavra(resultado[1], marca.localizacao, marca.linha,
                                             marca.linha_localizacao, marca.bloco)

        return dicio

    @staticmethod
    def corrigir_gravidade(dicio):
        gravidade = dicio['gravidade']
        gravidade_texto = gravidade.texto

        try:
            gravidade_num = int(gravidade_texto)

            if gravidade_num == 3:
                gravidade_texto = 'Leve'
            elif gravidade_num == 4:
                gravidade_texto = 'Média'
            elif gravidade_num == 5:
                gravidade_texto = 'Grave'
            elif gravidade_num == 7:
                gravidade_texto = 'Gravíssima'
            else:
                gravidade_texto = ''
        except ValueError:
            pass

        dicio['gravidade'] = Palavra(gravidade_texto, gravidade.localizacao, gravidade.linha,
                                     gravidade.linha_localizacao, gravidade.bloco)
        return dicio

    @staticmethod
    def determinar_pessoa(dicio):
        fisica = Palavra('Física', ((0, 0), (0, 0)), 0, ((0, 0), (0, 0)), 0)
        juridica = Palavra('Jurídica', ((0, 0), (0, 0)), 0, ((0, 0), (0, 0)), 0)
        vazio = Palavra('', ((0, 0), (0, 0)), 0, ((0, 0), (0, 0)), 0)

        proprietario_cpf = dicio['proprietario_cpf'].texto if \
            len(dicio['proprietario_cpf'].texto) != 0 else ''
        condutor_cpf = dicio['condutor_cpf'].texto if \
            len(dicio['condutor_cpf'].texto) != 0 else ''

        proprietario_cnpj = dicio['proprietario_cnpj'].texto if \
            len(dicio['proprietario_cnpj'].texto) != 0 else ''
        condutor_cnpj = dicio['condutor_cnpj'].texto if \
            len(dicio['condutor_cnpj'].texto) != 0 else ''

        if len(proprietario_cpf) > 0:
            dicio['proprietario_pessoa'] = fisica
        elif len(proprietario_cnpj) > 0:
            dicio['proprietario_pessoa'] = juridica
        else:
            dicio['proprietario_pessoa'] = vazio

        if condutor_cpf is not None and len(condutor_cpf) > 0:
            dicio['condutor_pessoa'] = fisica
        elif len(condutor_cnpj) > 0:
            dicio['condutor_pessoa'] = juridica
        else:
            dicio['condutor_pessoa'] = vazio

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
    def testar_valores(valor_1, valor_2):
        resultado = list()

        if len(valor_1) == 0:
            resultado = valor_2
        elif len(valor_2) == 0:
            resultado = valor_1
        else:
            resultado = valor_1 if len(valor_1) >= len(valor_2) else valor_2

        return resultado[0] if len(resultado) > 0 else Palavra('', ((0, 0), (0, 0)), 0, ((0, 0), (0, 0)), 0)
