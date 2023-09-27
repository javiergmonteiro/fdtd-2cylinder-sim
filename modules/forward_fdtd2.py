#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microwave imaging: solution for many forward problems

Este módulo contiene una función para calcular numéricamente
las imágenes de 16x16 formadas por lo recibido por cada
antena emitiendo con cada receptor. También guarda
un archivo con los parámetros que simuló.

Módulo Python: forward_problem
Author: Ramiro Irastorza
Email: rirastorza@iflysib.unlp.edu.ar

"""
import os
import random
import sys

import meep as mp
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as S

mp.verbosity.meep = 0
mp.verbosity.ctl = 0
mp.verbosity.mbp = 0
mp.simulation.do_progress = False

#
# - Constantes
#

pi = S.pi
eps0 = S.epsilon_0
c = S.c
mu0 = S.mu_0

a = 0.005  # Meep unit


#
# - Clases de parámetros del dispersor, acoplante, y transmisor.
#
class SCATTERER_parameters:
    # Cilindro
    epsr = 1.2  # permitividad relativa
    sigma = 0.0  # conductividad
    mur = 1.0
    f = 1.0e9  # frecuencia 1 GHz (por defecto).
    # epsrC = epsr + sigma/(2j*pi*f*eps0);
    radio = 0.25  # *c/f #radio del cilindro
    xc = 0.0  # 0.75*c/f #radio del cilindro
    yc = 0.0  # 0.75*c/f #radio del cilindro
    # k = 2*pi*f*((epsrC*mur)**0.5)/c


class ACOPLANTE_parameters:
    epsr = 1.0  # frecuencia 1 GHz (por defecto).
    sigma = 0.0  # conductividad
    mur = 1.0
    f = 1.0e9  # frecuencia 1 GHz (por defecto).


class TRANSMISOR_parameters:
    f = 1.0e9  # frecuencia 1 GHz (por defecto).
    rhoS = 0.075  # *c/f #radio de transmisores
    S = 16.  # cantidad de transmisores (fuentes)
    amp = 1000.0  # Amplitud de la fuente


#
# - Función para cambio de coordenadas cartesianas a polares
#
def cart2pol(x, y):
    rho = (x ** 2.0 + y ** 2.0) ** 0.5
    phi = np.arctan2(y, x)
    # phi = N.arctan(y/x)
    return phi, rho


#
# - Definición de funciones
#


#
# - Función numérica con FDTD utilizando software meep
#
def RunMeep2(cilindro1, cilindro2, acoplante, trans, Tx, caja, RES=5, calibration=False, unit=None):
    res = RES  # pixels/a
    dpml = 1

    sx = caja[0] / a
    sy = caja[1] / a

    # print('sxa: ', sx, 'sxa: ', sy)

    # rhoS = tran1.5*c/trans.f

    fcen = trans.f * (a / c)  # pulse center frequency
    sigmaBackgroundMeep = acoplante.sigma * a / (c * acoplante.epsr * eps0)
    sigmaCylinderMeep = cilindro1.sigma * a / (c * cilindro1.epsr * eps0)
    sigmaCylinderMeep2 = cilindro2.sigma * a / (c * cilindro2.epsr * eps0)

    materialBackground = mp.Medium(epsilon=acoplante.epsr,
                                   D_conductivity=sigmaBackgroundMeep)  # Background dielectric properties at operation frequency
    materialCilindro = mp.Medium(epsilon=cilindro1.epsr,
                                 D_conductivity=sigmaCylinderMeep)  # Cylinder dielectric properties at operation frequency
    materialCilindro2 = mp.Medium(epsilon=cilindro2.epsr,
                                  D_conductivity=sigmaCylinderMeep2)  # Cylinder dielectric properties at operation frequency

    default_material = materialBackground

    # Simulation box and elements
    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml)]

    if calibration:  # el cilindro1 del centro es Background
        geometry = [mp.Cylinder(material=materialBackground, radius=cilindro1.radio / a, height=mp.inf,
                                center=mp.Vector3(cilindro1.xc / a, cilindro1.yc / a, 0))]
    else:  # el cilindro1 del centro es la muestra
        geometry = [mp.Cylinder(material=materialCilindro, radius=cilindro1.radio / a, height=mp.inf,
                                center=mp.Vector3(cilindro1.xc / a, cilindro1.yc / a, 0)),
                    mp.Cylinder(material=materialCilindro2, radius=cilindro2.radio / a, height=mp.inf,
                                center=mp.Vector3(cilindro2.xc / a, cilindro2.yc / a, 0))]

    xt = (trans.rhoS) * np.cos(Tx * 2 * pi / trans.S)  # Coordenada x antena transmisora
    yt = (trans.rhoS) * np.sin(Tx * 2 * pi / trans.S)  # Coordenada y antena transmisora

    sources = [mp.Source(mp.ContinuousSource(frequency=fcen), component=mp.Ez, center=mp.Vector3(xt / a, yt / a, 0.0),
                         amplitude=trans.amp, size=mp.Vector3(0.0, 0.0, mp.inf))]

    sim = mp.Simulation(cell_size=cell, sources=sources, resolution=res, default_material=default_material,
                        eps_averaging=False, geometry=geometry, boundary_layers=pml_layers, force_complex_fields=True)

    nt = 600

    sim.run(until=nt)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    return ez_data, eps_data


def generate_cylinders_data():
    pass


def simulate():

    # Graficos
    landa = c / TRANSMISOR_parameters.f
    sx = 0.25
    sy = 0.25
    Tx = 0
    box = [sx, sy]

    TRANSMISOR_parameters.f = 1.1e9
    TRANSMISOR_parameters.amp = 7500.
    TRANSMISOR_parameters.rhoS = 0.075
    TRANSMISOR_parameters.S = 16.

    ACOPLANTE_parameters.f = 1.1e9
    ACOPLANTE_parameters.epsr = 28.6  # frecuencia 1 GHz (por defecto).
    ACOPLANTE_parameters.sigma = 1.264

    Ntotal_modelos = int(1e4)  # numero total de modelos
    semilla = int(87539319)  # La semilla
    np.random.seed(semilla)  # Seteo la semilla del generador

    # Coordenadas antenas
    angulo = np.linspace(0.0, 2.0 * pi, 17)
    xantenas = (TRANSMISOR_parameters.rhoS) * np.cos(angulo)
    yantenas = (TRANSMISOR_parameters.rhoS) * np.sin(angulo)

    # Propiedades del cilindro

    center_x = random.uniform(0)

    cilindro1 = SCATTERER_parameters()
    cilindro1.epsr = random.uniform(10.0, 80.0)  # permitividad relativa. Entre [10.0, 80.0]
    cilindro1.sigma = random.uniform(0.40, 1.60)  # conductividad. Entre [0.40, 1.60]

    cilindro1.f = 1.1e9  # frecuencia 1 GHz (por defecto).
    cilindro1.radio = 0.017215338370147
    cilindro1.xc = 0.017727534921579
    cilindro1.yc = 0.036450109026959

    cilindro2 = SCATTERER_parameters()
    cilindro2.epsr = 19.914118835371848  # permitividad relativa. Entre [10.0, 80.0]
    cilindro2.sigma = 1.297486619795625  # conductividad. Entre [0.40, 1.60]
    cilindro2.f = 1.1e9  # frecuencia 1 GHz (por defecto).
    cilindro2.radio = 0.017215338370147
    cilindro2.xc = 0.017727534921579
    cilindro2.yc = 0.036450109026959

    resolucion = 5
    n = resolucion * sx / a
    Tx = np.arange(16)
    Tr = np.arange(16)
    EzTr = np.zeros((16, 16))
    for tx in Tx:
        Ezfdtd, eps_data = RunMeep2(cilindro1, cilindro2, ACOPLANTE_parameters, TRANSMISOR_parameters, tx, box,
                                    calibration=False)
        Ezfdtdinc, eps_data_no = RunMeep2(cilindro1, cilindro2, ACOPLANTE_parameters, TRANSMISOR_parameters, tx, box,
                                          calibration=True)
        for tr in Tr:
            xSint = int(resolucion * ((0.15 / 2) * np.cos(tr * 2 * pi / 16.)) / a) + int(n / 2)  # Coordenada x antena emisora
            ySint = int(resolucion * ((0.15 / 2) * np.sin(tr * 2 * pi / 16.)) / a) + int(n / 2)
            EzTr[tx, tr] = abs(Ezfdtd)[xSint, ySint] / abs(Ezfdtdinc)[xSint, ySint]

        print('Campo en emisor:', EzTr[tx, tx])

    # Dibujo la imagen de entrada
    plt.figure()
    plt.imshow(EzTr, cmap='binary')
    plt.colorbar()
    plt.show()

    # Dibujo el mapa de permitividad
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.show()