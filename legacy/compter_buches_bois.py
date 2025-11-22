import cv2
import numpy as np

# --- Load image ---
img = cv2.imread("buches_cropped.png")
output = img.copy()

# --- Preprocessing ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

# --- Circle detection ---
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=80,
    param2=25,
    minRadius=15,
    maxRadius=60
)

log_count = 0

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        # Optional: remove circles too high / too low in the image (noise)
        if y < 200 or y > img.shape[0] - 50:
            continue

        log_count += 1
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

print("Detected logs:", log_count)

cv2.imwrite("buches_circles_cropped.jpg", output)
print("Saved: buches_circles_cropped.jpg")
