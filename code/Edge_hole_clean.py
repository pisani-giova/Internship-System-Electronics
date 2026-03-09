import cv2 as cv
import numpy as np
from collections import Counter
import math

# --------------------------
# PARAMETRI (da calibrare)
# --------------------------

MIN_RADIUS = 10
MAX_RADIUS = 100
THRESHOLD_FILL = 125
CANNY1=105
CANNY2=25

# --------------------------
# Funzioni
# --------------------------

def rescale(image, scale=0.05):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv.resize(image, dim)

def mostfrequent(vettore):
    conteggio = Counter(vettore)
    return conteggio.most_common(1)[0][0]

def count_near_circles(centri, centro_riferimento, raggio):
    count = 0
    x0, y0 = centro_riferimento
    x0 = int(x0)
    y0 = int(y0)

    r2 = float(raggio) ** 2

    for x, y in centri:
        x = int(x)
        y = int(y)

        dx = x - x0
        dy = y - y0
        d2 = dx*dx + dy*dy

        if dx == 0 and dy == 0:
            continue

        if d2 <= r2:
            count += 1

    return count

# --------------------------
# Main
# --------------------------

# --------------------------
# Rimozione Background
# --------------------------

img = cv.imread("../Photo/front_tray.jpeg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(7,7),0)

# -----------------------
# threshold
# -----------------------

_,thresh = cv.threshold(
    blur,
    0,
    255,
    cv.THRESH_BINARY_INV + cv.THRESH_OTSU
)

# -----------------------
# floodfill per riempire buchi
# -----------------------

h,w = thresh.shape
mask = np.zeros((h+2,w+2),np.uint8)

flood = thresh.copy()
cv.floodFill(flood,mask,(0,0),255)

flood_inv = cv.bitwise_not(flood)
tray_mask = thresh | flood_inv

# -----------------------
# chiude bordo dentato
# -----------------------

kernel = np.ones((25,25),np.uint8)

tray_mask = cv.morphologyEx(
    tray_mask,
    cv.MORPH_CLOSE,
    kernel
)

# -----------------------
# trova contorno tray
# -----------------------

contours,_ = cv.findContours(
    tray_mask,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

largest = max(contours, key=cv.contourArea)

rect = cv.minAreaRect(largest)
box = cv.boxPoints(rect)
box = np.intp(box)

cv.drawContours(img,[box],0,(0,255,0),4)

# -----------------------
# crop tray
# -----------------------

x,y,w,h = cv.boundingRect(largest)

# margine per ridurre il cropping aggressivo
margin = 30

x = max(0, x + margin)
y = max(0, y + margin)
w = min(img.shape[1] - x, w + 2*margin)
h = min(img.shape[0] - y, h + 2*margin)

tray = img[y:y+h, x:x+w]
cv.imshow("tray", tray)

gray_tray = cv.cvtColor(tray, cv.COLOR_BGR2GRAY)
gaus_tray = cv.GaussianBlur(gray_tray, (11, 11), 0)

# --------------------------
# Trovo raggio standard
# --------------------------

circles_standard = cv.HoughCircles(
    gaus_tray,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=CANNY1,
    param2=CANNY2,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS
)

vettrag=[]

if circles_standard is not None:
    circles = np.uint16(np.around(circles_standard))

    for (x, y, r) in circles[0,:]:
        vettrag.append(r)

standard=mostfrequent(vettrag)

# --------------------------
# Trova cerchi
# --------------------------

circles = cv.HoughCircles(
    gaus_tray,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=CANNY1,
    param2=CANNY2,
    minRadius=standard-20,
    maxRadius=standard+20
)

filled = 0
empty = 0
raggi=[]
centri=[]

if circles is not None:
    circles = np.uint16(np.around(circles))

    for (x, y, r) in circles[0,:]:

        mask = np.zeros(gaus_tray.shape, dtype=np.uint8)
        cv.circle(mask, (x, y), r-10, 255, -1)

        # --------------------------
        # controllo pieno o vuoto
        # --------------------------

        mean_val = cv.mean(gaus_tray, mask=mask)[0]
        print("Mean:", mean_val)

        # --------------------------
        # riempio i vettori con posizioni e raggi dei cerchi trovati
        # --------------------------
        raggi.append(r)
        centri.append((x,y))


        if mean_val > THRESHOLD_FILL:
            status = "FILLED"
            color = (0,255,0)
            filled += 1
        else:
            status = "EMPTY"
            color = (0,0,255)
            empty += 1

        cv.circle(tray, (x, y), r, color, 3)
        cv.putText(tray, status, (x-30, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

# --------------------------
#ricerca dei bordi
# --------------------------

bordi=[]
cont=0
r_medio = mostfrequent(raggi)
raggio_vicinanza = 2.2 * r_medio

for x, y in centri:
    n_vicini = count_near_circles(centri, (x, y), 170)


#2,4 per bordi su e giu 2,3 per destra e sinitra , 2,3,4 per tutti i buchi bordosi
    if n_vicini in (2,3,4):
        bordi.append((x, y))
        cont += 1
        cv.circle(tray, (x, y), r_medio, (255, 0, 0), 3)



# --------------------------
# Testo riassuntivo
# --------------------------

cv.putText(tray,
            f"Filled: {filled}  Empty: {empty}",
            (30,50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)

# --------------------------
# RIDIMENSIONAMENTO SOLO PER DISPLAY
# --------------------------

max_width = 1000  # larghezza massima finestra

h, w = tray.shape[:2]

if w > max_width:
    scale = max_width / (w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    display = cv.resize(tray, (new_w, new_h))
else:
    display = tray

# --------------------------
# Risultato
# --------------------------

cv.imshow("Result", display)
cv.waitKey(0)
cv.destroyAllWindows()