import cv2 as cv

def rescale(image, scale=0.05):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv.resize(image, dim)

img = cv.imread('../Photos/cestello.jpg')
resize = rescale(img)

gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
gauss = cv.GaussianBlur(gray, (9,9), 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
erode = cv.erode(gauss, kernel)

thresh = cv.adaptiveThreshold(erode, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
canny = cv.Canny(thresh, 50, 250)


contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

count = 0
for cnt in contours:
    area = cv.contourArea(cnt)
    print(f"Area: {area}")

    (x, y), radius = cv.minEnclosingCircle(cnt)
    if area >= 1000 and area < 5000:
        count += 1
        cv.circle(resize, (int(x), int(y)), int(radius), (0,0,255), 2)

print("Buchi: {count_buchi}")
cv.putText(resize, f"Buchi: {count}", (380, 25), cv.FONT_ITALIC, 1, (255, 0, 0), 3)



cv.imshow("risultato", thresh)
cv.imshow("detection", resize)


cv.waitKey()
cv.destroyAllWindows()