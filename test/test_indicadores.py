import unicodedata
import pandas as pd


def test_gerar_indicadores():
    caminho_comparacao = '../resultados.csv'
    l_percent, m_porcen = test_indicador_percentual_chave_preenchida(caminho_comparacao)
    l_resul, m_p = test_indicador_percentual_campo_correto(caminho_comparacao)


def test_indicador_percentual_chave_preenchida(caminho_comp):
    caminho = '../Base_de_Dados.csv'

    df_manual = pd.read_csv(caminho)
    df_programa = pd.read_csv(caminho_comp)

    valor_manual = nome_arquivo_numero_valores(df_manual)
    valor_programa = nome_arquivo_numero_valores(df_programa)

    lista_resultado = list()
    for t_1, t_2 in zip(valor_manual, valor_programa):
        lista_resultado.append((t_1[0], t_2[1] * 100 / t_1[1]))

    media_porcentagem = sum([val[1] for val in lista_resultado]) / len(lista_resultado)

    return lista_resultado, media_porcentagem


def test_indicador_chaves_encontradas(caminho_comp):
    caminho = '../Base_de_Dados.csv'

    df_manual = pd.read_csv(caminho)
    df_programa = pd.read_csv(caminho_comp)

    valor_manual = nome_arquivo_numero_valores(df_manual)
    valor_programa = nome_arquivo_numero_valores(df_programa)

    lista_resultado = list()
    for t_1, t_2 in zip(valor_manual, valor_programa):
        lista_resultado.append((t_1[0], t_2[1] * 100 / t_1[1]))

    media_porcentagem = sum([val[1] for val in lista_resultado]) / len(lista_resultado)

    return lista_resultado, media_porcentagem


def test_indicador_percentual_campo_correto(caminho_comp='../resultados.csv'):
    caminho = '../Base_de_Dados.csv'
    caminho_comp = '../resultados.csv'

    df_manual = pd.read_csv(caminho)
    df_programa = pd.read_csv(caminho_comp)

    resultado = list()
    for idx in range(0, df_manual.shape[0]):
        iguais = 0
        l_man = str(df_manual.iloc[idx].tolist())
        l_pro = str(df_programa.iloc[idx].tolist())

        for v_m, v_p in zip(l_man, l_pro):
            if pd.isna(v_m) and pd.isna(v_p):
                iguais += 1
            elif pd.isna(v_m) and not pd.isna(v_p):
                iguais += 0
            elif not pd.isna(v_m) and pd.isna(v_p):
                iguais += 0
            else:
                if e_igual(v_m, v_p):
                    iguais += 1

        resultado.append((int(l_man[1]), iguais * 100 / len(l_man)))

    media_porcentagem = sum([val[1] for val in resultado]) / len(resultado)

    return resultado, media_porcentagem


def nome_arquivo_numero_valores(df):
    resultado = list()
    for idx in range(0, df.shape[0]):
        lista_valores = df.iloc[idx].tolist()
        valores = len(list(filter(lambda val: not pd.isna(val), lista_valores))) - 1

        resultado.append((lista_valores[0], valores))

    return resultado


def e_igual(p1, p2):
    return unicodedata\
        .normalize('NFD', p1)\
        .encode('ascii', 'ignore')\
        .decode('utf-8')\
        .lower()\
        .strip()\
        .__contains__(p2.lower())
