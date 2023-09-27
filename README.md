# Simulador de images de campo electrico para cilindros mediante el método FDTD.


### Introducción

Esta herramienta genera datos simulados de la geometría de dos cilindros:
uno dentro de otro y tambien sus características diélectricas, esto se guarda un archivo
.csv con el nombre de "output.csv". Este script guardará imagenes de 16x16 del campo electrico
generado por un arreglo de 16 antenas al rededor de estos cilindros almacenandolas en un archivo
de numpy llamado "input.npy". Adicionalmente se podrán grabar las imagenes en formato .png
si asi lo desea siguiendo los pasos del script.

### Requerimientos
- Miniconda con meep 1.26

### Ejecución
- Ejecutar el archivo "main.py" y seguir los pasos