# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:04:42 2023

@author: Daniela Cuartas
"""
import cv2
import numpy as np
import pytesseract
import re
from collections import Counter
from skimage import morphology
from scipy import ndimage

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Esta funcion es una herramienta para establecer los valores del bounding box de deteccion (verde)
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Posición (x, y): ({x}, {y}) - Valor del píxel (B, G, R): {frame[y, x]}")

def filtrar_cadena(cadena):
    # Utilizar una expresión regular para encontrar solo caracteres alfanuméricos en mayúsculas y el guion
    cadena_filtrada = re.sub(r'[^A-Z0-9]', '', cadena)
    return cadena_filtrada

def erase_objects(imagen, area_threshold=50):
    # Invierte los colores
    cierre = 255 - imagen
    
    # Encuentra los contornos en la imagen
    contours, _ = cv2.findContours(cierre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra los contornos por área
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]
    
    # Crea una imagen en blanco del mismo tamaño que la imagen original
    result = np.zeros_like(cierre)
    
    # Dibuja los contornos filtrados en la nueva imagen
    cv2.drawContours(result, filtered_contours, -1, 255, thickness=cv2.FILLED)
    
    # Invierte nuevamente los colores
    result = 255 - result
    
    return result

#----------- Captura de video ----------------------
cap = cv2.VideoCapture("parqueadero2.mp4")

Ctexto = ""
output_ocr = []
count = 0

window_name = 'Vehiculos'
bin_window_name = 'Binarizado'
cv2.namedWindow(bin_window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
desired_width = 800   # Puedes ajustar a tu preferencia
desired_height = 600  # Puedes ajustar a tu preferencia
cv2.resizeWindow(window_name, desired_width, desired_height)
cv2.resizeWindow(bin_window_name, desired_width//2, desired_height//2)

# cv2.setMouseCallback(bin_window_name, mouse_callback) # Comentar cuando no se use
paused = False
while True:
    if not paused:
        ret, frame = cap.read()
    
        if ret == False:
            break
        
        # se extrae el ancho y alto de los fotogramas
        alto, ancho, c = frame.shape
        
        # En x:    
        x1 = int(670)
        x2 = int(1700)
        
        # En y:
        y1 = int(400)
        y2 = int(1050)
        
        # Texto
        cv2.putText(frame, "Procesando placa", (320, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # se ubica el rectangulo en las posiciones extraidas
        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
        
        recorte = frame[y1:y2, x1:x2]
        # se realiza el recorte de la zona de interes
        redc = np.matrix(recorte[:,:,2])
        bluec = np.matrix(recorte[:,:,0])
        greenc = np.matrix(recorte[:,:,1])
        
        # color
        color = cv2.absdiff(greenc, bluec)
        
        # se binariza la imagen
        # _, umbral = cv2.threshold(color, 80, 255, cv2.THRESH_BINARY)
        _, umbral = cv2.threshold(color, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow(bin_window_name, umbral)
        
        # se extraen los contornos de la zona seleccionada
        contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # se ordenan del mas grande al mas pequeño
        contornos = sorted(contornos, key=lambda x:cv2.contourArea(x), reverse=True)
        
        # se dibujan los contornos extraidos
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 2000 and area < 15000:
                # se detecta la placa
                x, y, ancho, alto = cv2.boundingRect(contorno)
                
                # se extraen las coordenadas de la placa
                xpi = x + x1
                ypi = y + y1
                
                xpf = x + ancho + x1
                ypf = y + alto + y1
                
                # se dibuja el rectangulo de la placa
                cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255,255,0), 2)
                
                # Recorte de la PLACA
                placa = frame[ypi:ypf, xpi:xpf]
                # cv2.imshow(bin_window_name, placa)
                # cv2.imwrite('placa.jpg', placa)
                
                # %%
                gray_img = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
                # cv2.imshow(bin_window_name, gray_img)
                
                blur_img = cv2.GaussianBlur(gray_img, (5,5), 0)
                # cv2.imshow(bin_window_name, blur_img)
                
                _,umbralito = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cv2.imshow(bin_window_name, umbralito)  
                
                
                # Realizar la operación de apertura
                ee = np.ones((3,3), np.uint8)
                # apertura = cv2.morphologyEx(umbralito, cv2.MORPH_OPEN, ee)
                # cv2.imshow(bin_window_name, apertura)
                
                # Realizar la operación de cierre
                cierre = cv2.morphologyEx(umbralito, cv2.MORPH_CLOSE, ee)
                
                # cierre = erase_objects(cierre, 80)
                
                cv2.imshow(bin_window_name, cierre)
                
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # contrast_enhanced_image = clahe.apply(umbralito)
                
                
                # cv2.imwrite('placa_bw.jpg', apertura)
                # %%
                
                # se extraen el ancho y el alto de los fotogramas
                altop, anchop, cp = placa.shape
                # print(altop, anchop)
                
                # se verifica que el tamaño de la placa sea lo suficientemente grande
                if altop >= 20 and anchop >= 60:
                    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesserac.exe'
                    # se extrae el texto
                    config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    texto = pytesseract.image_to_string(cierre, config= config, output_type=pytesseract.Output.STRING)
    
                    if len(texto) >= 7:
                        # Ctexto = texto
                        output_ocr.append(texto)
                        count += 1
                        # print(Ctexto)
                
                # Sistema de votacion
                if count >= 8:
                    conteo_frecuencias = Counter(output_ocr)
                    resultado_final = conteo_frecuencias.most_common(1)[0][0]
                    print("Resultado de OCR más frecuente:", resultado_final)
                    Ctexto = filtrar_cadena(resultado_final)
                    count = 0
                    output_ocr = []
                    
        
        
        # Dibujar rectangulo donde va la placa
        cv2.rectangle(frame, (350,355), (600,400), (0,0,0), cv2.FILLED)
        cv2.putText(frame, Ctexto, (390,390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Mostrar el frame con el rectángulo
        cv2.imshow(window_name, frame)
        
    key = cv2.waitKey(1) & 0xFF
    # Si se presiona la tecla 'q', salir del bucle
    if key == ord('q'):
        break
    # Si se presiona la tecla 'barra espaciadora', pausar video
    elif key == 32:
        paused = not paused
        
# Libera la captura de la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()