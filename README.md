# Detección de placas 

Se realiza la extracción y detección de caracteres en placas de dos formas diferentes, por medio de métodos clásicos de algoritmos de procesamiento digital de imágenes, y con el uso de YOLOV8.

## Usando procesamiento digital de imágenes 
La imagen muestra el proceso realizado para la extracción y deteccion de caracteres en las placas.
El archivo lectura_placa_v4.py, contiene el código necesario para la ejecución de este. 
NOTA: es importante asegurarse de cambiar el archivo de video cargado por el tuyo, a la hora de ejecutar el código.
![image](https://github.com/dani-cuar/Plates-detection/assets/42179443/35023cfa-85a1-4032-b503-a2387413d3ed)


## Usando YOLOV8

Para el entrenamiento de YOLO, se hizo uso de la base de datos pública de la universidad popular del Cesar, que puedes encontrar en [Roboflow](https://universe.roboflow.com/universidad-popular-del-cesar-pmj7r/prueba_1-tnlwa)

![image](https://github.com/dani-cuar/Plates-detection/assets/42179443/5cf6df35-ee52-4504-a3b2-1df6c3bb28e8)
El archivo yolo_detection.py contiene el código necesario para la ejecución de este.
