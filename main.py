import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
import speech_recognition as sr

# --- Inicializar MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Variables del ratón ---
sensitivity = 1.0
last_click_time = 0
click_cooldown = 0.3
prev_index = None

# --- Reconocimiento de voz ---
def voice_control():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Control por voz activo. Habla...")
        while True:
            try:
                audio = r.listen(source)
                text = r.recognize_google(audio, language="es-ES")
                print("Has dicho:", text)
                pyautogui.typewrite(text)
                pyautogui.press("enter")
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("⚠️ Error con el servicio de voz")

threading.Thread(target=voice_control, daemon=True).start()

# --- Función distancia ---
def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# --- Control de manos ---
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Coordenadas índice y pulgar
                index = (int(hand_landmarks.landmark[8].x * w),
                         int(hand_landmarks.landmark[8].y * h))
                thumb = (int(hand_landmarks.landmark[4].x * w),
                         int(hand_landmarks.landmark[4].y * h))

                # Dibujar puntos
                cv2.circle(frame, index, 10, (0, 255, 0), -1)
                cv2.circle(frame, thumb, 10, (0, 0, 255), -1)

                # Pinch: índice + pulgar juntos
                pinch_distance = distancia(index, thumb)
                if pinch_distance < 40:
                    cv2.putText(frame, "MODO CONTROL", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    if prev_index is not None:
                        dx = (index[0] - prev_index[0]) * sensitivity
                        dy = (index[1] - prev_index[1]) * sensitivity
                        pyautogui.moveRel(dx, dy, duration=0)  # movimiento inmediato
                    prev_index = index
                else:
                    prev_index = None
                    cv2.putText(frame, "MODO LIBRE", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Detectar puño cerrado
                finger_tips = [8, 12, 16, 20]
                folded = 0
                for tip in finger_tips:
                    tip_y = hand_landmarks.landmark[tip].y
                    pip_y = hand_landmarks.landmark[tip - 2].y
                    if tip_y > pip_y:
                        folded += 1

                if folded == 4 and (time.time() - last_click_time) > click_cooldown:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, "CLICK", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Control gestual + voz", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
