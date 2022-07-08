import glob
import argparse
from entidades.informações_recurso import InformacoesRecurso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--caminho', required=True,
                    help='Caminho da(s) imagem(ns)')
    ap.add_argument('-p', '--pasta', default=0,
                    help='Determina se deve selecionar todos os arquivos da pasta')
    ap.add_argument('-tm', '--teste-mascara', default=0,
                    help='Produz as máscaras binárias')
    ap.add_argument('-iou', '--medir-iou', default=0,
                    help='Salva um arquivo csv com valores do IOU')
    ap.add_argument('-sp', '--salvar-palavras', default=0,
                    help='Realiza a captura das palavras')
    ap.add_argument('-ai', '--ajustar-imagens', default=0,
                    help='Ajusta a imagem de acordo com o ROI escolhido pelo usuário')
    ap.add_argument('-do', '--desempenho-ocr', default=0,
                    help='Salva arquivo com os valores relevantes as palavras')
    ap.add_argument('-mc', '--marcar-chaves', default=0,
                    help='Marca palavras do formulário nas imagens')
    args = vars(ap.parse_args())

    arquivos = []

    if args['pasta']:
        caminho = args['caminho'] if args['caminho'].endswith('/') else args['caminho'] + '/'
        extensoes = ('*.jpeg', '*.jpg', '*.png')
        for ext in extensoes:
            arquivos.extend(glob.glob(caminho + '*' + ext))
    else:
        arquivos.append(args['caminho'])

    if args['teste_mascara']:
        for arquivo in arquivos:
            info_recurso = InformacoesRecurso()
            info_recurso(arquivo, teste_mascara=True)
    elif args['medir_iou']:
        info_recurso = InformacoesRecurso()
        info_recurso.iou('imagens/notificacoes_teste/', True, True)
    elif args['ajustar_imagens']:
        for arquivo in arquivos:
            info_recurso = InformacoesRecurso()
            info_recurso(arquivo, teste_ajuste_roi=True)
    elif args['desempenho_ocr']:
        info_recurso = InformacoesRecurso()
        info_recurso.desempenho_ocr()
    elif args['marcar_chaves']:
        info_recurso = InformacoesRecurso()
        info_recurso.marcar_chaves()
    elif args['salvar_palavras']:
        resultado = dict()

        for arquivo in arquivos:
            nome_arquivo = arquivo.split('/')[-1]
            info_recurso = InformacoesRecurso()
            palavras = info_recurso(arquivo)
            resultado[nome_arquivo] = {'palavras': palavras}

        info_recurso.salvar_palavras(resultado, 'fixtures/', 'palavras_teste_final')
    else:
        for arquivo in arquivos:
            info_recurso = InformacoesRecurso()
            info_recurso(arquivo)


main()
