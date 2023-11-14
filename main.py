import csv
import os
import queue
import sys
import threading
import time
import concurrent.futures

import numpy as np
from matplotlib import pyplot as plt

from modules.generate_data import generate_models

base_dir = os.getcwd()
pid = os.getpid()

print("process pid " + str(pid))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def test_func(thread_list, iterations):
    print("Comenzando simulación...")
    print("Estimando tiempo de espera...")
    single = 136
    total_estimated_time = single * iterations
    print("Tiempo estimado de: {} minutos".format(round(total_estimated_time / 60, 2)))
    start = time.time()
    while True:
        now = time.time()
        percentage = (int(now - start) * 100) / total_estimated_time
        if percentage < 100:
            print("##########################   Porcentaje completado: {}   #############################".format(round(percentage, 2)))
        if "RUNNING" not in [t._state for t in thread_list] and percentage >= 50:
            finished = int(time.time() - start)
            print("Simulación terminada con {} segundos".format(finished))
            break
        else:
            if percentage >= 100:
                print("#############################   Tiempo estimado completo, terminando ultimas simulaciones...  ###################################")
        time.sleep(5)


if __name__ == '__main__':

    threads = []
    # results = []
    result_queue = queue.Queue()

    print("Ingrese la cantidad de simulaciones a realizar: ")
    simulations = int(input())
    print("¿Desea guardar las imagenes (formato .png) (s/n)?:")
    save_images = str(input())

    # for i in range(simulations):
    #     thread = threading.Thread(target=generate_models, args=(result_queue,))
    #     # thread = threading.Thread(target=test_func, args=(i, result_queue, ))
    #     threads.append(thread)
    #
    # control_thread = threading.Thread(target=test_func, args=(threads, simulations,))
    # control_thread.start()
    #
    # for thread in threads:
    #     thread.start()
    #
    # for thread in threads:
    #     thread.join()
    #
    # # Collect results from the queue
    # while not result_queue.empty():
    #     results.append(result_queue.get())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit the function calls with different parameters
        futures = [executor.submit(generate_models, result_queue) for _ in range(simulations)]  # sim times
        control_thread = threading.Thread(target=test_func, args=(futures, simulations,))
        control_thread.start()

    # Retrieve the results when they are ready
    results = [future.result() for future in futures]

    print("Guardando resultados...")
    file_directory = base_dir + '/output.csv_{}'.format(pid)
    csvfile = open(file_directory, 'w', newline='')
    fieldnames = ['radius', 'xc', 'yc', 'iradius', 'ixc', 'iyc', 'epsilon', 'sigma', 'iepsilon', 'isigma']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    cyl_params = [d[1] for d in results]
    writer.writeheader()
    writer.writerows(cyl_params)
    csvfile.close()
    data_input = [d[0] for d in results]
    np.save(base_dir + '/input.npy_{}'.format(pid), np.array(data_input, dtype=object), allow_pickle=True)
    print("input y output han sido guardados")

    if save_images.lower() == 's':
        print("Guardando imagenes..")
        input_data = np.load(base_dir + '/input.npy_{}'.format(pid), allow_pickle=True)
        input_data = np.asarray(input_data).astype('float32')
        # print(results)
        for idx, idata in enumerate(input_data):
            figure, axes = plt.subplots()
            plt.xlim(-0.25 / 2, 0.25 / 2)
            plt.ylim(-0.25 / 2, 0.25 / 2)
            plt.figure()
            plt.imshow(idata, cmap='binary')
            plt.colorbar()
            plt.savefig('EzTr {0}.png'.format(idx))
            plt.close()
        print("Imagenes guardadas")
