import cv2

rtp_url = "udp://@127.0.0.1:5005"
cap = cv2.VideoCapture(rtp_url)

if not cap.isOpened():
    print("Error: cannot open video stream.")
else:

    # Lire et afficher le flux vidéo
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: cannot receive frame.")
            break

        # Afficher la frame
        cv2.imshow('UDP stream from MuJoCo', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
