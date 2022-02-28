import cv2 as cv2
import imutils
import numpy as np


def retirar_sombras(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.split(rgb)

    planos_norm_resultantes = list()
    for plano in rgb:
        img_dilatada = cv2.dilate(plano, np.ones((7, 7), np.uint8))
        img_blur = cv2.medianBlur(img_dilatada, 21)
        img_diff = 255 - cv2.absdiff(plano, img_blur)
        img_norm = cv2.normalize(img_diff, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        img_norm = cv2.morphologyEx(img_norm, cv2.MORPH_ERODE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        img_norm = cv2.morphologyEx(img_norm, cv2.MORPH_DILATE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        planos_norm_resultantes.append(img_norm)

    resultado_norm = cv2.merge(planos_norm_resultantes)

    return cv2.cvtColor(resultado_norm, cv2.COLOR_RGB2BGR)
