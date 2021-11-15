import argparse
from pdf2image import convert_from_path

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--caminho', type=str, required=True,
                help='Caminho do arquivo de PDF')
ap.add_argument('-n', '--nome', type=str, required=True,
                help='Nome do arquivo de saída')
args = vars(ap.parse_args())

pages = convert_from_path(args['caminho'], 500)

for page in pages:
    page.save(args['nome'] + '.png', 'PNG')
