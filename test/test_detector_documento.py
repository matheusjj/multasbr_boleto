import pytest
from detectores.detector_documento import DetectorDocumentoHough


def test_angulo_entre_pontos():
    detector = DetectorDocumentoHough()
    angulo = detector.angulo_entre_pontos([[[1, 1]], [[-1, -1]]])
    assert(angulo, 180)
