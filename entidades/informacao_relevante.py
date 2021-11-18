import statistics


class InformacaoRelevante:
    def __init__(self, texto, num_linha, num_palavra, topo, esquerda, largura):
        self.texto = texto.strip()
        self.num_linha = num_linha
        self.num_palavra = num_palavra
        self.topo = topo
        self.esquerda = esquerda
        self.largura = largura
        self.direita = esquerda + largura

    @staticmethod
    def retirar_vazios(lista) -> list:
        return list(filter(lambda val: len(val.texto) > 0, lista))

    @staticmethod
    def selecionar_mesma_linha(lista) -> list:
        linha = statistics.mode([x.num_linha for x in lista])
        return list(filter(lambda val: val.num_linha == linha, lista))

    @staticmethod
    def determinar_frases(lista, posicao_assunto):
        frases = {}
        tamanho_espaco = 20
        e_frase = False
        chave_frase_atual = None

        for (idx, palavra) in enumerate(lista):
            if idx == len(lista) - 1:
                if e_frase:
                    frases[chave_frase_atual].append(palavra)
                else:
                    frases[palavra.esquerda] = [palavra]
                break

            distancia = lista[idx + 1].esquerda - palavra.direita
            if not e_frase and distancia < tamanho_espaco:
                e_frase = True
                chave_frase_atual = palavra.esquerda
                frases[chave_frase_atual] = [palavra]
            elif not e_frase and distancia > tamanho_espaco:
                frases[palavra.esquerda] = [palavra]
            elif e_frase and distancia < tamanho_espaco:
                frases[chave_frase_atual].append(palavra)
            elif e_frase and distancia > tamanho_espaco:
                e_frase = False
                frases[chave_frase_atual].append(palavra)

        resultados = []
        for (idx, val) in enumerate(list(frases.keys())):
            resultados.append((idx, (val / posicao_assunto) - 1))

        minimo = min([x[1] for x in resultados], key=abs)
        resultado_final = list(filter(lambda result: result[1] == minimo, resultados))[0]
        chave_final = list(frases.keys())[resultado_final[0]]

        frase_final = ''
        for palavra in frases[chave_final]:
            frase_final += "{} ".format(palavra.texto)

        return frase_final.strip()
