import cv2
import numpy as np
from keras_facenet import FaceNet
from numpy import expand_dims
from os import listdir, makedirs
from os.path import join, isdir, exists
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pywhatkit
import time
import datetime
import smtplib
from email.message import EmailMessage
from gtts import gTTS
import playsound

# Constants
LOG_FILE = "security_log.txt"
INTRUDER_FOLDER = "Intruders"
EMAIL_ADDRESS = "your_email@example.com"
EMAIL_PASSWORD = "your_email_password"
PHONE_NUMBER = "+91XXXXXXXXXX"

# Initialize FaceNet and HaarCascade
HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
FaceNetModel = FaceNet()

# Ensure Intruder folder exists
if not exists(INTRUDER_FOLDER):
    makedirs(INTRUDER_FOLDER)

def log_event(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} {message}\n")

def save_intruder(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = join(INTRUDER_FOLDER, f"intruder_{timestamp}.jpg")
    cv2.imwrite(file_path, image)
    log_event(f"\U0001F6A8 Intruder detected! Image saved as {file_path}")
    return file_path

def send_alert():
    try:
        # Using pywhatkit to send a WhatsApp message without opening the web window
        pywhatkit.sendwhatmsg_instantly(PHONE_NUMBER, ' ALERT: Unrecognized person detected!', tab_close=True)
        print("WhatsApp alert sent!")
        log_event("WhatsApp alert sent.")
    except Exception as e:
        print(f"WhatsApp alert failed: {e}")

def send_email_alert(image_path):
    msg = EmailMessage()
    msg["Subject"] = "Security Alert: Unrecognized Person Detected!"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = "security_team@example.com"
    msg.set_content("An unrecognized person was detected. See attached image.")
    with open(image_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="intruder.jpg")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email alert sent!")
        log_event("Email alert sent.")
    except Exception as e:
        print(f"Email alert failed: {e}")

def play_warning():
    tts = gTTS("Warning! You are not authorized to enter!", lang="en")
    tts.save("warning.mp3")
    playsound.playsound("warning.mp3")

def get_all_face_embeddings(image):
    faces = HaarCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)
    embeddings = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face).resize((160, 160))
        face = np.asarray(face)
        face = expand_dims(face, axis=0)
        embeddings.append(FaceNetModel.embeddings(face))
    return embeddings

def create_database(base_folder):
    database = {}
    for person in listdir(base_folder):
        person_folder = join(base_folder, person)
        if isdir(person_folder):
            embeddings = []
            for img in listdir(person_folder):
                img_path = join(person_folder, img)
                image = cv2.imread(img_path)
                face_embeddings = get_all_face_embeddings(image)
                
                if face_embeddings:  # Ensure it's not empty
                    embeddings.extend(face_embeddings)  # Store all embeddings

            if embeddings:  # Ensure at least one valid embedding exists
                try:
                    database[person] = np.mean(np.array(embeddings), axis=0)
                except ValueError as e:
                    print(f"Error processing {person}: {e}")
                    log_event(f"Error processing {person}: {e}")

    return database

def identify_face(database, embedding, threshold=0.6):
    for person, stored_embedding in database.items():
        similarity = cosine_similarity(embedding, stored_embedding)[0][0]
        if similarity > threshold:
            return person
    return None

database = create_database("/Users/asujan23/Main Project/code")
camera = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
print("🎥 Monitoring for motion... Press 'q' to exit.")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50:
            frame = gray
        fgmask = fgbg.apply(frame)
        motion_detected = np.sum(fgmask) > 500000
        if motion_detected:
            print("🚨 Motion detected! Scanning for faces...")
            embeddings = get_all_face_embeddings(frame)
            if embeddings:
                authorized_found = False
                for embedding in embeddings:
                    name = identify_face(database, embedding)
                    if name:
                        print(f"{name} is authorized.")
                        log_event(f"Authorized: {name}")
                        authorized_found = True
                        break
                if not authorized_found:
                    print("Unrecognized face detected! Sending alert...")
                    img_path = save_intruder(frame)
                    send_alert()  # WhatsApp alert without window
                    send_email_alert(img_path)
                    play_warning()  # Play warning sound
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting monitoring...")
            break
except KeyboardInterrupt:
    print("\nStopped manually.")
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera released. Monitoring stopped.")
