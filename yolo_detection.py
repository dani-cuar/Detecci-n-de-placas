import os
import time
import numpy as np
import re
import torch

# Configurar la variable de entorno
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import cv2

import string
#import easyocr
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Initialize the OCR reader
#reader = easyocr.Reader(['en'], gpu=False)

def filtrar_cadena(cadena):
    # Utilizar una expresión regular para encontrar solo caracteres alfanuméricos en mayúsculas y el guion
    cadena_filtrada = re.sub(r'[^A-Z0-9]', '', cadena)
    return cadena_filtrada

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    config = r'--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    #detections = reader.readtext(license_plate_crop)
    text = pytesseract.image_to_string(license_plate_crop, config= config, output_type=pytesseract.Output.STRING)
    
    text=filtrar_cadena(text)
    if len(text) >= 6:
        return text
    """
    for detection in detections:
        bbox, text, score = detection
        return text[0:7]"""

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Posición (x, y): ({x}, {y}) - Valor del píxel (B, G, R): {frame[y, x]}")


text = ""
bin_window_name = 'recorte'
cv2.namedWindow(bin_window_name, cv2.WINDOW_NORMAL)

bin_window_name_2 = 'bin'
cv2.namedWindow(bin_window_name_2, cv2.WINDOW_NORMAL)

desired_width = 800   # Puedes ajustar a tu preferencia
desired_height = 600  # Puedes ajustar a tu preferencia

cv2.resizeWindow(bin_window_name, desired_width//2, desired_height//2)
cv2.resizeWindow(bin_window_name_2, desired_width//2, desired_height//2)

window_name = 'Vehiculos'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, desired_width, desired_height)
x1_=450
x2_=1000
y1_= 200
y2_=700
y1=600


# load models

# Check if GPU is available
if torch.cuda.is_available():
    # Get the current GPU device index
    license_plate_detector = YOLO('best.pt').to('cuda')
    current_gpu = torch.cuda.current_device()
    print(f"Currently using GPU {current_gpu}")
else:
    license_plate_detector = YOLO('best.pt')
    print("GPU is not available. Using CPU.")

# load video
cap = cv2.VideoCapture('parqueadero2.mp4')
#cv2.setMouseCallback(window_name, mouse_callback)
# read frames
text=None
while True:
    #inicio = time.time()
    ret, frame = cap.read()
    # bounding box detection
    
    # cv2.rectangle(frame, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
    if ret: 
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        #print("license plate", license_plates)

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            #print("coordenadas: ", x1,y1,x2,y2)

            # if True:#x1 >= x1_ and y1 >= y1_ and x2 <= x2_ and y2 <= y2_:
            # crop license plate
            license_plate_crop = frame[int(y1):int(y2)+15, int(x1): int(x2), :]

            # Dibujar bounding box en el frame original
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)+15), (0, 255, 0), 2)

            cv2.imshow(bin_window_name, license_plate_crop)

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            blur_img = cv2.GaussianBlur(license_plate_crop_gray, (5,5), 0)
            _, license_plate_crop_thresh = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            ee = np.ones((3,3), np.uint8)
            # Realizar la operación de cierre
            cierre = cv2.morphologyEx(license_plate_crop_thresh, cv2.MORPH_CLOSE, ee)

            cv2.imshow(bin_window_name_2, cierre)

            # read license plate number
            license_plate_text = read_license_plate(cierre)
            if license_plate_text:
                text = filtrar_cadena(license_plate_text)
                #print(text)

        # Mostrar el frame original con bounding box y texto
        text_position = (10, 30)  # Coordenadas del texto en la esquina superior izquierda
        if text:
            cv2.putText(frame, f"License Plate: {text}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)
    #fin = time.time()
    #print("tiempo total: ", fin-inicio)
    # Esperar 1 milisegundo y verificar si se presiona una tecla
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Libera la captura de la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()
