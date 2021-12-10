from entidades.palavra import Palavra
from extratores.extrator_informacao import IERegrado
from fixtures.ex_1 import lista_palavras
from fixtures.ex_2 import lista_palavras_2
from fixtures.ex_3 import lista_palavras_3
from fixtures.ex_4 import lista_palavras_4


def test_separar_em_frases():
    palavras = [
        Palavra('Hoje', ((1, 1), (2, 1)), 2, 1),
        Palavra('é', ((3, 1), (4, 1)), 2, 1),
        Palavra('um', ((5, 1), (6, 1)), 2, 1),
        Palavra('dia', ((30, 1), (31, 1)), 2, 1),
        Palavra('de', ((32, 1), (33, 1)), 2, 1),
        Palavra('um', ((70, 1), (71, 1)), 2, 1),
        Palavra('novo', ((100, 1), (101, 1)), 2, 1),
        Palavra('tempo', ((130, 1), (131, 1)), 2, 1),
        Palavra('que', ((132, 1), (133, 1)), 2, 1),
        Palavra('começou', ((170, 1), (171, 1)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.separar_em_frases(palavras)
    assert resultado == {}


def test_separar_em_frases():
    palavras = [
        Palavra('Hoje', ((1, 1), (2, 1)), 2, 1),
        Palavra('é', ((30, 1), (31, 1)), 2, 1),
        Palavra('um', ((60, 1), (61, 1)), 2, 1),
        Palavra('dia', ((62, 1), (63, 1)), 2, 1),
        Palavra('de', ((90, 1), (91, 1)), 2, 1),
        Palavra('um', ((92, 1), (93, 1)), 2, 1),
        Palavra('novo', ((94, 1), (95, 1)), 2, 1),
        Palavra('tempo', ((130, 1), (131, 1)), 2, 1),
        Palavra('que', ((132, 1), (133, 1)), 2, 1),
        Palavra('começou', ((170, 1), (171, 1)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.separar_em_frases(palavras)
    assert resultado == {}


def test_separar_em_frases():
    palavras = [
        Palavra('Hoje', ((1, 1), (2, 1)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.separar_em_frases(palavras)
    assert resultado == {}


def test_separar_em_frases():
    palavras = [
        Palavra('Hoje', ((1, 1), (2, 1)), 2, 1),
        Palavra('é', ((3, 1), (4, 1)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.separar_em_frases(palavras)
    assert resultado == {}


def test_separar_em_frases():
    palavras = [
        Palavra('Hoje', ((1, 1), (2, 1)), 2, 1),
        Palavra('é', ((30, 1), (31, 1)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.separar_em_frases(palavras)
    assert resultado == {}


def test_determinar_conjunto_proximo():
    locais_assunto = [
        Palavra('proprietario', ((5, 1), (6, 1)), 2, 1),
    ]
    locais_campo = [
        Palavra('nome', ((5, 10), (6, 11)), 2, 1),
        Palavra('nome', ((5, 20), (6, 21)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.determinar_conjunto_proximo(locais_assunto, locais_campo)
    assert resultado == {}


def test_determinar_conjunto_proximo():
    locais_assunto = [
        Palavra('proprietario', ((5, 1), (6, 1)), 2, 1),
    ]
    locais_campo = [
        Palavra('nome', ((5, 20), (6, 21)), 2, 1),
        Palavra('nome', ((5, 10), (6, 11)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.determinar_conjunto_proximo(locais_assunto, locais_campo)
    assert resultado == {}


def test_determinar_conjunto_proximo():
    locais_assunto = [
        Palavra('proprietario', ((5, 1), (6, 1)), 2, 1),
        Palavra('proprietario', ((5, 15), (6, 16)), 2, 1),
    ]
    locais_campo = [
        Palavra('nome', ((5, 20), (6, 21)), 2, 1),
        Palavra('nome', ((5, 10), (6, 11)), 2, 1),
    ]
    extrator = IERegrado()
    resultado = extrator.determinar_conjunto_proximo(locais_assunto, locais_campo)
    assert resultado == {}


def test_extrator_informacao_ex1():
    palavras = [Palavra(p[0], p[1], p[2], p[3], p[4]) for p in lista_palavras]

    extrator = IERegrado()
    extrator(palavras, [])
    print('oi')


def test_extrator_informacao_ex2():
    palavras = [Palavra(p[0], p[1], p[2], p[3], p[4]) for p in lista_palavras_2]

    extrator = IERegrado()
    extrator(palavras, [])
    print('oi')


def test_extrator_informacao_ex3():
    palavras = [Palavra(p[0], p[1], p[2], p[3], p[4]) for p in lista_palavras_3]

    extrator = IERegrado()
    extrator(palavras, [])
    print('oi')


def test_extrator_informacao_ex4():
    palavras = [Palavra(p[0], p[1], p[2], p[3], p[4]) for p in lista_palavras_4]

    extrator = IERegrado()
    extrator(palavras, [])
    print('oi')
