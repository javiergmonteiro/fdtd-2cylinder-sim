#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from modules.forward_fdtd2 import ACOPLANTE_parameters, TRANSMISOR_parameters, pi, SCATTERER_parameters, a, RunMeep2
from modules.classes import Antenna, Cylinder, DataWriter
import numpy as np
import time as tm


def generate_models(queue=None):
    """
    Función principal encargada de generar todos los datos de la simulación, estos son:
    - Los valores de entrada que representan EzTr.
    - Los valores de salida que representan los valores geométricos y dielétricos de los cilindors: radios, centros(x,y)
    permitividad y conductividad

    :return:
    :return:
        List()
        - EzTr: NumpyArray() Matriz 16x16 que representa la imagen del campo electrico
        - cil_data: Dict() Diccionario que contiene la clave valor de cada elemento descrito anteriormente.
    """
    # Graficos
    sx = 0.25
    sy = 0.25
    box = [sx, sy]

    antenna = Antenna()

    r = np.random.uniform(0.01, 0.040)
    anglescat = np.random.uniform(0.0, 2 * pi)
    dext = np.random.uniform(0.0, TRANSMISOR_parameters.rhoS - 1e-2 - r)
    Xc = dext * np.cos(anglescat)
    Yc = dext * np.sin(anglescat)

    rin = np.random.uniform(0.005, r * 0.95)
    anglescatin = np.random.uniform(0.0, 2 * pi)
    d = np.random.uniform(0.0, r - rin)
    Xcin = Xc + d * np.cos(anglescatin)
    Ycin = Yc + d * np.sin(anglescatin)
    cilindros = Cylinder(antenna, r, Xc, Yc, rin, Xcin, Ycin)

    # cilindros.draw()

    # # Comienzo de simulación
    #
    # cilindro1 = SCATTERER_parameters()
    # cilindro1.epsr = 19.914118835371848  # permitividad relativa. Entre [10.0, 80.0]
    # cilindro1.sigma = 1.297486619795625  # conductividad. Entre [0.40, 1.60]
    #
    # cilindro1.f = 1.1e9  # frecuencia 1 GHz (por defecto).
    # cilindro1.radio = 0.022649387788057
    # cilindro1.xc = 0.015902529364148  # 0.017727534921579
    # cilindro1.yc = 0.032697661106962  # 0.036450109026959
    #
    # cilindro2 = SCATTERER_parameters()
    # cilindro2.epsr = 19.914118835371848  # permitividad relativa. Entre [10.0, 80.0]
    # cilindro2.sigma = 1.297486619795625  # conductividad. Entre [0.40, 1.60]
    # cilindro2.f = 1.1e9  # frecuencia 1 GHz (por defecto).
    # cilindro2.radio = 0.02
    # cilindro2.xc = 0.015902529364148  # 0.017727534921579
    # cilindro2.yc = 0.032697661106962  # 0.036450109026959

    # print('a: ', a)

    resolucion = 5
    n = resolucion * sx / a
    Tx = np.arange(16)
    Tr = np.arange(16)
    EzTr = np.zeros((16, 16))
    for tx in Tx:
        Ezfdtd, eps_data = RunMeep2(cilindros[0], cilindros[1], ACOPLANTE_parameters, TRANSMISOR_parameters, tx, box,
                                    calibration=False, unit=0.005)
        Ezfdtdinc, eps_data_no = RunMeep2(cilindros[0], cilindros[1], ACOPLANTE_parameters, TRANSMISOR_parameters, tx,
                                          box,
                                          calibration=True, unit=0.005)
        xFuente_int = int(resolucion * ((0.15 / 2) * np.cos(tx * 2 * pi / 16.)) / a) + int(
            n / 2)  # Coordenada x antena emisora
        yFuente_int = int(resolucion * ((0.15 / 2) * np.sin(tx * 2 * pi / 16.)) / a) + int(n / 2)

        for tr in Tr:
            xSint = int(resolucion * ((0.15 / 2) * np.cos(tr * 2 * pi / 16.)) / a) + int(
                n / 2)  # Coordenada x antena receptora
            ySint = int(resolucion * ((0.15 / 2) * np.sin(tr * 2 * pi / 16.)) / a) + int(n / 2)
            EzTr[tx, tr] = abs(Ezfdtd)[xSint, ySint] / abs(Ezfdtdinc)[xSint, ySint]

        EzTr[tx, :] = np.roll(EzTr[tx, :], -tx)

    cil_data = cilindros.as_dict()
    if queue:
        queue.put((EzTr, cil_data))
    return EzTr, cil_data


if __name__ == '__main__':

    print("Introduzca la cantidad de imagenes a generar:")
    count = int(input())
    print("Introduzca la semilla (deje vacio para usar por defecto):")
    seed = input()
    if not seed:
        seed = int(87539319)
    else:
        seed = int(seed)
    np.random.seed(seed)
    start_time = tm.strftime('%H:%M:%S')
    print('start time: ', start_time)
    data_writer = DataWriter(save_images=True)
    for j in range(0, count):
        input, output = generate_models()
        data_writer.write_input(input)
        data_writer.write_output(output)
    data_writer.close()
    print('end time:   ', tm.strftime('%H:%M:%S'))
