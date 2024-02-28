# Detección de placas 

Instala el archivo de requerimientos "requeriments.txt" con el comando pip install requirements.txt.

Se realiza la extracción y detección de caracteres en placas de dos formas diferentes, por medio de métodos clásicos de algoritmos de procesamiento digital de imágenes, y con el uso de YOLOV8.

## Usando procesamiento digital de imágenes 
La imagen muestra el proceso realizado para la extracción y deteccion de caracteres en las placas.

El archivo lectura_placa_v4.py, contiene el código necesario para la ejecución de este. 

NOTA: es importante asegurarse de cambiar el archivo de video cargado por el tuyo, a la hora de ejecutar el código.

No olvides cambiar el path de "pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'" por el path donde tengas tu librería instalada.

![image](https://github.com/dani-cuar/Plates-detection/assets/42179443/35023cfa-85a1-4032-b503-a2387413d3ed)


## Usando YOLOV8

Para el entrenamiento de YOLO, se hizo uso de la base de datos pública de la universidad popular del Cesar, que puedes encontrar en [Roboflow](https://universe.roboflow.com/universidad-popular-del-cesar-pmj7r/prueba_1-tnlwa)

![image](https://github.com/dani-cuar/Plates-detection/assets/42179443/5cf6df35-ee52-4504-a3b2-1df6c3bb28e8)

El archivo yolo_detection.py contiene el código necesario para la ejecución de este.

Recuerda cargar el archivo de pesos generado luego de ejecutar el entrenamiento del modelo en YOLOV8_plate_detection.ipynb, asi como cargar el respectivo archivo mp4 de acuerdo a tu video.

NOTA: es importante contar con una GPU a la hora de ejecutar el código, esto facilitará su visualización y ejecución, ya que de otra forma el código avanza demasiado lento, o incluso podría no funcionar.

No olvides cambiar el path de "pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'" por el path donde tengas tu librería instalada.
