import csv
import os
import shutil
import sys
import numpy as np
from modules.forward_fdtd2 import ACOPLANTE_parameters, TRANSMISOR_parameters, pi, plt


def draw_ef_image(data):
    plt.figure()
    plt.imshow(data, cmap='binary')
    plt.colorbar()
    plt.show()


# Clases

class DataWriter:
    """
    Clase encargada de volcar en disco los valores de la simulación, por defecto estos son:
    - Para la entrada (input) que representan las imagenes de 16x16 en formato Numpy Array (npy) con el nombre de input.npy
    tambien se puede optar adicionalmente para guardar las imagenes en formato .png.
    - Para la salida (output) que representan los datos de los cilindros y sus propiedades (etiquetas) en formato .csv
    """

    def _raise_critical_error(self, exception):
        print("No se pudieron crear los directorios para los datos. Detalle de error: {}".format(exception))
        sys.exit()

    def __init__(self, save_images=False):
        try:
            data_dirs = ['/train', '/validation', '/test']
            self.base_directories = list()

            for d in data_dirs:
                try:
                    base_dir = os.getcwd() + d
                    self.base_directories.append(base_dir)
                    os.mkdir(base_dir)
                except FileExistsError:
                    pass
        except Exception as e:
            self._raise_critical_error(e)

        self.fieldnames = ['radius', 'xc', 'yc', 'iradius', 'ixc', 'iyc', 'epsilon', 'sigma', 'iepsilon', 'isigma']
        self.save_images = save_images
        self.data_input = list()
        try:
            self.file_directory = self.base_directories[0] + '/output-data.csv'
            self.csvfile = open(self.file_directory, 'w', newline='')
            self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
            self.writer.writeheader()
        except Exception as e:
            self._raise_critical_error(e)

    def write_output(self, data):
        self.writer.writerows([data])

    def write_input(self, data):
        self.data_input.append(data)

    def close(self):
        for dir in self.base_directories:
            try:
                shutil.copy(self.file_directory, dir + '/output-data.csv')
            except shutil.SameFileError:
                pass
        for dir in self.base_directories:
            np.save(dir + '/input.npy', np.array(self.data_input, dtype=object), allow_pickle=True)
            # b = np.load('input.npy', allow_pickle=True)
        if self.save_images:
            for idx, idata in enumerate(self.data_input):
                figure, axes = plt.subplots()
                plt.xlim(-0.25 / 2, 0.25 / 2)
                plt.ylim(-0.25 / 2, 0.25 / 2)
                plt.figure()
                plt.imshow(idata, cmap='binary')
                plt.colorbar()
                image = 'input_{}.png'.format(idx)
                for dir in self.base_directories:
                    plt.savefig(dir + '/{}'.format(image))
                plt.close()
        self.csvfile.close()


class Antenna:
    """
    Esta clase representa al arreglo de antenas en su conjunto y contiene
    los valores simulados captados a partir de las propiedades dieléctricas
    """

    def __init__(self):
        self.TRANSMISOR_parameters = TRANSMISOR_parameters()
        self.TRANSMISOR_parameters.f = 1.1e9
        self.TRANSMISOR_parameters.amp = 7.5e4
        self.TRANSMISOR_parameters.rhoS = 0.075
        self.TRANSMISOR_parameters.S = 16.

        self.ACOPLANTE_parameters = ACOPLANTE_parameters()
        self.ACOPLANTE_parameters.f = 1.1e9
        self.ACOPLANTE_parameters.epsr = 28.6  # frecuencia 1 GHz (por defecto).
        self.ACOPLANTE_parameters.sigma = 1.264

        # self.Ntotal_modelos = 1  # numero total de modelos

        # Propiedades dieléctricas
        self.epsext = np.random.uniform(10.0, 80.0)
        self.sigext = np.random.uniform(0.40, 1.60)
        self.epsin = np.random.uniform(10.0, 80.0)
        self.sigin = np.random.uniform(0.40, 1.60)

        # Coordenadas antenas
        self.angulo = np.linspace(0.0, 2.0 * pi, 17)
        self.xantenas = self.TRANSMISOR_parameters.rhoS * np.cos(self.angulo)
        self.yantenas = self.TRANSMISOR_parameters.rhoS * np.sin(self.angulo)


class Cylinder:
    """
    El siguiente objeto contiene la información de los atributos
    y las propiedades dieléctricas de dos cilindros uno interno y otro externo.

    Atributos:
        radius: radio del cilindro externo.
        xc: centro en el plano x del cilindro externo.
        yc: centro en el plano y del cilindo externo.
        epsilon: permitividad relativa del cilindro externo.
        sigma: conductividad del cilindro externo.
        iradius: radio del cilindro interno.
        ixc: centro en el plano x del cilindro interno.
        iyc: centro en el plano y del cilindo interno.
        iepsilon: permitividad relativa del cilindro interno.
        isigma: conductividad del cilindro interno.

    """

    def __init__(self, antenna, radius, xc, yc, iradius, ixc, iyc, epsilon=None, sigma=None, iepsilon=None,
                 isigma=None):
        self.antenna = antenna
        self.radius = radius
        self.xc = xc
        self.yc = yc
        self.iradius = iradius
        self.ixc = ixc
        self.iyc = iyc
        if self.antenna:
            self.epsilon = antenna.epsext
            self.sigma = antenna.sigext
            self.iepsilon = antenna.epsin
            self.isigma = antenna.sigin
        else:
            self.epsilon = epsilon
            self.sigma = sigma
            self.iepsilon = iepsilon
            self.isigma = isigma

    def format(self, separation='comma'):
        if separation == 'comma':
            return "{},{},{},{},{},{},{},{},{},{}".format(
                self.radius,
                self.xc,
                self.yc,
                self.iradius,
                self.ixc,
                self.iyc,
                self.epsilon,
                self.sigma,
                self.iepsilon,
                self.isigma
            )

    def draw(self, save=False, index=None):
        figure, axes = plt.subplots()
        plt.xlim(-0.25 / 2, 0.25 / 2)
        plt.ylim(-0.25 / 2, 0.25 / 2)
        cilindroext = plt.Circle((self.xc, self.yc), self.radius, fill=False)
        cilindroin = plt.Circle((self.ixc, self.iyc), self.iradius)
        axes.set_aspect(1)
        axes.add_artist(cilindroext)
        axes.add_artist(cilindroin)
        if self.antenna:
            axes.plot(self.antenna.xantenas, self.antenna.yantenas, 'ok')
        # if save and index is not None:
        #     image_directories = [dir for dir in directories if 'images' in dir]
        #     for idir in image_directories:
        #         plt.savefig(idir + '/cylinder_{}.png'.format(index))
        plt.show()

    def as_dict(self):
        data = self.__dict__.copy()
        del data['antenna']
        return data

    def __getitem__(self, item):
        class CIL:
            pass
        cil = CIL()
        if item == 0:
            cil.radio = self.radius
            cil.xc = self.xc
            cil.yc = self.yc
            cil.epsr = self.epsilon
            cil.sigma = self.sigma
            cil.mur = 1.0
            cil.f = 1.0e9
            return cil
        elif item == 1:
            cil.radio = self.iradius
            cil.xc = self.ixc
            cil.yc = self.iyc
            cil.epsr = self.iepsilon
            cil.sigma = self.isigma
            cil.mur = 1.0
            cil.f = 1.0e9
            return cil
        return None



