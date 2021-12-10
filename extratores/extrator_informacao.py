import re
import unicodedata


class IERegrado:
    def __init__(self):
        self.e_pesquisa_vertical = None
        self.cpf_re = re.compile(r"\d{3}[.]?\d{3}[.]?\d{3}[-]?\d{2}")
        self.cnpj_re = re.compile(r"\d{2}[.]?\d{3}[.]?\d{3}[/]?\d{4}[-]?\d{2}")
        self.placa_re = re.compile(r"\D{3}[-]?\d{4}|\D{3}\d\D\d{2}")
        self.estado_re = re.compile(r"[A-Z]{2}\b")
        # self.estado_re = re.compile(r"\b[I]?[A-Z]{2}")
        self.estados_brasileiros = ['RR', 'AP', 'AM', 'PA', 'AC', 'RO', 'TO', 'MA', 'PI', 'CE', 'RN',
                                    'PB', 'PE', 'AL', 'SE', 'BA', 'MT', 'DF', 'GO', 'MS', 'MG', 'ES',
                                    'RJ', 'SP', 'PR', 'SC', 'RS']
        self.informacoes_relevantes = {
            'proprietario': {'nome': None, 'cpf': None, 'cnpj': None, 'pessoa': None},
            'condutor': {'nome': None, 'cpf': None, 'cnpj': None, 'pessoa': None},
            'veiculo': {'placa': None, 'marca': None, 'renavam': None},
            'local': {'uf': None},
            'uf': None,
            # 'categoria': None,
            # 'natureza': None,
        }

    def __call__(self, palavras, img):
        self.palavras = palavras
        self.img = img

        for info_primaria in self.informacoes_relevantes:
            if type(self.informacoes_relevantes[info_primaria]) is dict:
                for info_secundaria in self.informacoes_relevantes[info_primaria]:
                    self.buscar_informacoes(info_primaria, info_secundaria)
            else:
                self.buscar_informacoes(info_primaria)

    def buscar_informacoes(self, info_primaria, info_secundaria=None):
        self.e_pesquisa_vertical = True
        # self.e_pesquisa_vertical = False
        self.pesquisa(info_primaria, info_secundaria)

    def pesquisa(self, assunto, campo=None):
        locais_assunto = self.filtrar_palavras_encontradas(assunto)

        if campo is not None:
            locais_campo = self.filtrar_palavras_encontradas(campo)

            if len(locais_campo) != 0:
                locais_palavras_interesse = self.determinar_conjunto_proximo(locais_assunto, locais_campo)
                self.preenche_informacao(locais_palavras_interesse, assunto, campo)

            return

        if len(locais_assunto) != 0:
            locais_palavras_interesse = {i: locais_assunto[i] for i in range(0, len(locais_assunto))}
            self.preenche_informacao(locais_palavras_interesse, assunto)

    def preenche_informacao(self, locais_interesse, assunto, campo=None):
        if campo is None:
            frases_selecionadas = self.coletar_resultados(locais_interesse, assunto, assunto)
        else:
            frases_selecionadas = self.coletar_resultados(locais_interesse, campo, campo)

        if len(frases_selecionadas) != 0:
            frases_selecionadas = frases_selecionadas[0]
            resultado = ''
            for palavra in frases_selecionadas:
                resultado += palavra.texto + ' '

            resultado = resultado.strip()
            if campo is None:
                resultado = self.extrair_palavras(assunto, resultado)
                self.informacoes_relevantes[assunto] = resultado
            else:
                resultado = self.extrair_palavras(campo, resultado)
                self.informacoes_relevantes[assunto][campo] = resultado

    def coletar_resultados(self, locais, campo, campo_verificar):
        frases_selecionadas = list()

        for campo in locais.values():
            if self.e_pesquisa_vertical:
                palavras_proxima_linha = list(filter(lambda palavra:
                                                     palavra.linha == campo.linha + 1
                                                     and palavra.bloco == campo.bloco,
                                                     self.palavras))
            else:
                palavras_proxima_linha = list(filter(lambda palavra:
                                                     palavra.linha == campo.linha
                                                     and palavra.bloco == campo.bloco
                                                     and palavra.localizacao[0] != campo.localizacao[0],
                                                     self.palavras))

            palavras_proxima_linha = self.verificar_palavras(campo_verificar, palavras_proxima_linha)

            if len(palavras_proxima_linha) == 0:
                continue

            frases = self.separar_em_frases(palavras_proxima_linha)

            temp = []
            for idx, frase in enumerate(frases):
                temp.append((idx, abs((frase[0].localizacao[0][0] / campo.localizacao[0][0]) - 1)))

            temp.sort(key=lambda val: val[1])
            frases_selecionadas.append(frases[temp[0][0]])

        return frases_selecionadas

    def verificar_palavras(self, campo, palavras):
        if campo == 'cpf':
            return list(filter(lambda val: self.cpf_re.search(val.texto), palavras))
        elif campo == 'cnpj':
            return list(filter(lambda val: self.cnpj_re.search(val.texto), palavras))
        elif campo == 'placa':
            return list(filter(lambda val: self.placa_re.search(val.texto), palavras))
        elif campo == 'uf':
            return list(filter(lambda val: self.estado_re.search(val.texto), palavras))
        else:
            return palavras

    def extrair_palavras(self, campo, texto):
        if campo == 'cpf':
            return self.cpf_re.search(texto).group()
        elif campo == 'cnpj':
            return self.cnpj_re.search(texto).group()
        elif campo == 'placa':
            return self.placa_re.search(texto).group()
        elif campo == 'uf':
            resultado = list(filter(lambda val:
                                    val in self.estados_brasileiros,
                                    self.estado_re.findall(texto)))

            if len(resultado) == 0:
                return None
            elif len(resultado) > 1:
                return resultado[-1]
            else:
                return resultado[0]

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
