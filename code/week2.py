import cv2
import numpy as np
import time
from picamera2 import Picamera2
import math

# -----------------------
# PARAMETRI
# -----------------------

FRAME_WIDTH = 1200
FRAME_HEIGHT = 780

BLUR_KERNEL = (7, 7)

CIRCLE_MIN_DIST = 40
CIRCLE_PARAM1 = 110
CIRCLE_PARAM2 = 28
CIRCLE_MIN_RADIUS = 20
CIRCLE_MAX_RADIUS = 30

FILL_THRESHOLD = 65

PROCESS_EVERY = 2
FRAME_DELAY = 0

TRAY_REFRESH = 30
MAX_TRACK_POINTS = 50

# -----------------------
# STATO GLOBALE
# -----------------------

tray_roi = None
tray_locked = False

prev_gray = None
track_points = None

frame_counter = 0

# -----------------------
# CAMERA
# -----------------------

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
)

picam2.configure(config)
picam2.start()

# -----------------------
# FUNZIONI
# -----------------------

def ordina_punti_matrice(lista_pos, tolleranza_y=10):

    if not lista_pos:
        return []

    punti = sorted(lista_pos, key=lambda p: (p[1], p[0]))

    righe = []
    riga_corrente = [punti[0]]

    for p in punti[1:]:

        if abs(p[1] - riga_corrente[0][1]) <= tolleranza_y:
            riga_corrente.append(p)
        else:
            righe.append(riga_corrente)
            riga_corrente = [p]

    righe.append(riga_corrente)

    for i in range(len(righe)):
        righe[i] = sorted(righe[i], key=lambda p: p[0])

    ordinati = []
    for riga in righe:
        ordinati.extend(riga)

    return ordinati


def calcola_rotazione(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    dx = int(x2) - int(x1)
    dy = int(y2) - int(y1)

    angolo_rad = math.atan2(dy, dx)

    return math.degrees(angolo_rad)


# -----------------------
# LOOP PRINCIPALE
# -----------------------

try:

    while True:

        frame = picam2.capture_array()
        frame_counter += 1

        if frame_counter % PROCESS_EVERY != 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------
        # DETECT TRAY
        # -----------------------

        if not tray_locked or frame_counter % TRAY_REFRESH == 0:

            blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 1.5)
            eq = cv2.equalizeHist(blur)

            circles = cv2.HoughCircles(
                eq,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=CIRCLE_MIN_DIST,
                param1=CIRCLE_PARAM1,
                param2=CIRCLE_PARAM2,
                minRadius=CIRCLE_MIN_RADIUS,
                maxRadius=CIRCLE_MAX_RADIUS
            )

            if circles is not None:

                circles = np.uint16(np.around(circles[0]))

                if len(circles) >= 10:

                    xs = [int(c[0]) for c in circles]
                    ys = [int(c[1]) for c in circles]
                    rs = [int(c[2]) for c in circles]

                    min_x = min(xs)
                    max_x = max(xs)
                    min_y = min(ys)
                    max_y = max(ys)

                    avg_r = int(np.mean(rs))
                    margin = avg_r * 3

                    x = max(min_x - margin, 0)
                    y = max(min_y - margin, 0)

                    w = (max_x - min_x) + margin * 2
                    h = (max_y - min_y) + margin * 2

                    tray_roi = (x, y, w, h)
                    tray_locked = True

                    roi_img = gray[y:y+h, x:x+w]

                    track_points = cv2.goodFeaturesToTrack(
                        roi_img,
                        maxCorners=MAX_TRACK_POINTS,
                        qualityLevel=0.01,
                        minDistance=10
                    )

                    if track_points is not None:
                        track_points[:, 0, 0] += x
                        track_points[:, 0, 1] += y

                    prev_gray = gray.copy()

        # -----------------------
        # TRACKING
        # -----------------------

        if tray_locked and track_points is not None and prev_gray is not None:

            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                track_points,
                None
            )

            if new_pts is not None:

                good_new = new_pts[status == 1]
                good_old = track_points[status == 1]

                if len(good_new) >= 5:

                    movement = good_new - good_old

                    dx = np.mean(movement[:, 0])
                    dy = np.mean(movement[:, 1])

                    x, y, w, h = tray_roi

                    x += int(dx)
                    y += int(dy)

                    tray_roi = (x, y, w, h)

                    track_points = good_new.reshape(-1, 1, 2)

        prev_gray = gray.copy()

        # -----------------------
        # PROCESS TRAY
        # -----------------------

        if tray_locked:

            x, y, w, h = tray_roi

            img_h, img_w = gray.shape

            x = max(0, x)
            y = max(0, y)

            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w > 0 and h > 0:

                tray_img = gray[y:y+h, x:x+w]

                blur = cv2.GaussianBlur(tray_img, BLUR_KERNEL, 1.5)
                eq = cv2.equalizeHist(blur)

                circles = cv2.HoughCircles(
                    eq,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=CIRCLE_MIN_DIST,
                    param1=CIRCLE_PARAM1,
                    param2=CIRCLE_PARAM2,
                    minRadius=CIRCLE_MIN_RADIUS,
                    maxRadius=CIRCLE_MAX_RADIUS
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles[0]))

                results = []
                results2 = []

                filled = 0
                empty = 0

                if circles is not None:

                    for cx, cy, r in circles:

                        mask = np.zeros(tray_img.shape, np.uint8)
                        cv2.circle(mask, (cx, cy), int(r*0.7), 255, -1)

                        mean = cv2.mean(tray_img, mask=mask)[0]

                        if mean < FILL_THRESHOLD:

                            status = "EMPTY"
                            empty += 1
                            letter = "E"
                            color = (0,0,255)

                        else:

                            status = "FILLED"
                            filled += 1
                            letter = "F"
                            color = (0,200,0)

                        results.append((cx, cy, r, status))
                        results2.append((cx, cy))

                        x_draw = int(cx + x)
                        y_draw = int(cy + y)

                        cv2.circle(output,(x_draw,y_draw),int(r),color,3)
                        cv2.circle(output,(x_draw,y_draw),int(r*0.7),color,1)

                        cv2.circle(output,(x_draw,y_draw),3,(255,255,255),-1)

                        cv2.putText(
                            output,
                            letter,
                            (x_draw-8,y_draw+8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255,255,255),
                            2
                        )

                results2 = ordina_punti_matrice(results2,20)

                if len(results2) > 6:

                    pendenza = calcola_rotazione(results2[1], results2[3])

                    cv2.rectangle(output,(880,10),(1180,70),(0,0,0),-1)

                    cv2.putText(
                        output,
                        f"ROTATION: {round(pendenza,2)} deg",
                        (900,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,255),
                        2
                    )

                cv2.rectangle(output,(20,20),(330,90),(0,0,0),-1)

                cv2.putText(
                    output,
                    f"FILLED: {filled}",
                    (30,55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                cv2.putText(
                    output,
                    f"EMPTY: {empty}",
                    (180,55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2
                )

                cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,255),2)

                center_x = x + w//2
                center_y = y + h//2

                cv2.circle(output,(center_x,center_y),6,(255,255,0),-1)
                cv2.circle(output,(center_x,center_y),14,(255,255,0),1)

                cv2.line(output,(x,y),(x+w,y+h),(255,255,0),1)
                cv2.line(output,(x+w,y),(x,y+h),(255,255,0),1)

        cv2.imshow("result", output)

        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == ord('r'):
            tray_locked = False

        time.sleep(FRAME_DELAY)

finally:

    picam2.stop()
    cv2.destroyAllWindows()
