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

GRID_SCALE_X = 0.75
GRID_SCALE_Y = 0.75

BLUR_KERNEL = (9, 9)

CIRCLE_MIN_DIST = 60
CIRCLE_PARAM1 = 90
CIRCLE_PARAM2 = 22
CIRCLE_MIN_RADIUS = 30
CIRCLE_MAX_RADIUS = 38

FILL_THRESHOLD = 80

PROCESS_EVERY = 2
FRAME_DELAY = 0

TRAY_SCALE_FACTOR = 1.15
TRAY_RADIUS_MARGIN = 2.0

ROWS = 8
COLS = 8

frame_counter = 0

# -----------------------
# FUNZIONI
# -----------------------

def ordina_punti_matrice(lista_pos, tolleranza_y=10):

    if not lista_pos:
        return []

    punti = sorted(lista_pos, key=lambda p:(p[1],p[0]))

    righe=[]
    riga_corrente=[punti[0]]

    for p in punti[1:]:

        if abs(p[1]-riga_corrente[0][1])<=tolleranza_y:
            riga_corrente.append(p)
        else:
            righe.append(riga_corrente)
            riga_corrente=[p]

    righe.append(riga_corrente)

    for i in range(len(righe)):
        righe[i]=sorted(righe[i], key=lambda p:p[0])

    ordinati=[]
    for r in righe:
        ordinati.extend(r)

    return ordinati


def trova_rotazione_generale(lista_punti):

    if len(lista_punti) < 2:
        return 0.0

    punti = np.array(lista_punti, dtype=np.float64)

    centro = np.mean(punti, axis=0)
    punti_centrati = punti - centro

    cov = np.cov(punti_centrati.T)

    autovalori, autovettori = np.linalg.eig(cov)

    indice_max = np.argmax(autovalori)

    dx, dy = autovettori[:, indice_max]

    angolo = math.degrees(math.atan2(dy, dx))

    while angolo > 90:
        angolo -= 180

    while angolo < -90:
        angolo += 180

    return angolo


def remove_shadows(gray):

    gray = gray.astype(np.float32) + 1.0

    blur = cv2.GaussianBlur(gray,(0,0),80)

    retinex = np.log(gray) - np.log(blur)

    retinex = cv2.normalize(retinex,None,0,255,cv2.NORM_MINMAX)

    return retinex.astype(np.uint8)


# -----------------------
# GENERA GRIGLIA ANCORATA AL TRAY
# -----------------------

def ordina_griglia_reale(points):

    pts = np.array(points, dtype=np.float32)

    # calcolo rotazione
    ang = trova_rotazione_generale(points)
    rad = math.radians(-ang)

    cx = np.mean(pts[:,0])
    cy = np.mean(pts[:,1])

    rot=[]

    for x,y in pts:

        xr = cx + (x-cx)*math.cos(rad) - (y-cy)*math.sin(rad)
        yr = cy + (x-cx)*math.sin(rad) + (y-cy)*math.cos(rad)

        rot.append((xr,yr))

    rot = np.array(rot)

    # ordina per Y
    idx = np.argsort(rot[:,1])
    rot = rot[idx]
    pts = pts[idx]

    rows=[]
    current=[0]

    for i in range(1,len(rot)):

        if abs(rot[i][1]-rot[current[0]][1]) < 35:
            current.append(i)
        else:
            rows.append(current)
            current=[i]

    rows.append(current)

    ordered=[]

    for r in rows:

        row = pts[r]

        row = sorted(row, key=lambda p:p[0])

        ordered.extend(row)

    return ordered


def trova_vicino(p, detected, tol=40):

    for d in detected:

        dist = np.hypot(p[0]-d[0], p[1]-d[1])

        if dist < tol:
            return d

    return None


# -----------------------
# CAMERA
# -----------------------

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size":(FRAME_WIDTH,FRAME_HEIGHT),"format":"BGR888"}
)

picam2.configure(config)
picam2.start()


# -----------------------
# LOOP PRINCIPALE
# -----------------------

try:

    while True:

        frame = picam2.capture_array()

        frame_counter += 1

        if frame_counter % PROCESS_EVERY != 0:
            continue

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        output = frame.copy()

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        norm = remove_shadows(gray)

        blur=cv2.GaussianBlur(norm,BLUR_KERNEL,1.5)

        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

        eq=clahe.apply(blur)

        # -----------------------
        # DETECT CERCHI
        # -----------------------

        circles = cv2.HoughCircles(
            eq,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=CIRCLE_MIN_DIST,
            param1=CIRCLE_PARAM1,
            param2=CIRCLE_PARAM2,
            minRadius=CIRCLE_MIN_RADIUS,
            maxRadius=CIRCLE_MAX_RADIUS
        )

        if circles is None:

            cv2.imshow("result",output)

            if cv2.waitKey(1)==27:
                break

            continue

        circles = np.uint16(np.around(circles[0]))

        if len(circles) < 10:
            continue


        # -----------------------
        # TROVA TRAY
        # -----------------------

        pts = np.array([(c[0],c[1]) for c in circles],dtype=np.float32)

        hull = cv2.convexHull(pts)

        rect = cv2.minAreaRect(hull)

        (cx,cy),(w,h),angle = rect

        avg_r = np.mean([c[2] for c in circles])

        margin = avg_r * TRAY_RADIUS_MARGIN

        w = w * TRAY_SCALE_FACTOR + margin
        h = h * TRAY_SCALE_FACTOR + margin

        rect_expanded = ((cx,cy),(w,h),angle)

        box = cv2.boxPoints(rect_expanded)
        box = np.int32(box)

        cv2.drawContours(output,[box],0,(0,255,255),2)

        cx=int(cx)
        cy=int(cy)

        cv2.circle(output,(cx,cy),6,(255,255,0),-1)
        cv2.circle(output,(cx,cy),14,(255,255,0),1)

        # -----------------------
        # ANALISI RIEMPIMENTO
        # -----------------------

        filled=0
        empty=0

        detected_points=[]

        for x,y,r in circles:
            detected_points.append((x,y))

        points = [(c[0],c[1]) for c in circles]

        grid = ordina_griglia_reale(points)

        for gx,gy in grid:

            vicino = trova_vicino((gx,gy), detected_points)

            if vicino is not None:

                x,y = vicino

                mask=np.zeros(gray.shape,np.uint8)

                cv2.circle(mask,(x,y),int(avg_r*0.7),255,-1)

                mean=cv2.mean(gray,mask=mask)[0]

                if mean < FILL_THRESHOLD:

                    empty+=1
                    color=(0,0,255)
                    letter="E"

                else:

                    filled+=1
                    color=(0,200,0)
                    letter="F"

                draw_x,draw_y=x,y

            else:

                empty+=1
                color=(0,0,255)
                letter="E"
                draw_x,draw_y=gx,gy

            cv2.circle(output,(draw_x,draw_y),int(avg_r),color,2)

            cv2.circle(output,(draw_x,draw_y),3,(255,255,255),-1)

            cv2.putText(
                output,
                letter,
                (draw_x-8,draw_y+8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

        # -----------------------
        # ROTAZIONE
        # -----------------------

        if len(detected_points)>6:

            ang = trova_rotazione_generale(detected_points)

            cv2.rectangle(output,(880,10),(1180,70),(0,0,0),-1)

            cv2.putText(
                output,
                f"ROTATION: {round(ang,2)} deg",
                (900,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

        # -----------------------
        # UI (IDENTICA ALLA TUA)
        # -----------------------

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

        cv2.imshow("result",output)

        key=cv2.waitKey(1)

        if key==27:
            break

        time.sleep(FRAME_DELAY)

finally:

    picam2.stop()
    cv2.destroyAllWindows()
