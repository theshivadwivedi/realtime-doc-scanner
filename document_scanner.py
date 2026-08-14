import numpy as np
import cv2
import os
from datetime import datetime

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCORE_THRESHOLD = 15       
MIN_AREA_RATIO = 0.15      
MIN_EXTENT = 0.75          
MIN_BRIGHTNESS = 120       
MAX_SATURATION = 70        

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernal = np.ones((5,5), np.uint8)

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernal
    )

    return closed



def order_points(points):

    ordered = np.zeros((4,2), dtype="float32")

    s = points.sum(axis=1)
    ordered[0] = points[np.argmin(s)]
    ordered[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]

    return ordered

def cal_doc_size(ordered):
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])

    width = max(top_width, bottom_width)
    height = max(left_height, right_height)

    return width, height

def cal_score(area, height, width, contour_center, image_center):
    area_score = min(area/50000,10)

    aspect_ratio = width/height

    if 0.3 <= aspect_ratio <= 3.0:
        shape_score = 10
    else:
        shape_score = 0

    distance = np.linalg.norm(contour_center- image_center)

    max_distance = np.linalg.norm(image_center)

    center_score = max(0,10 -(distance/max_distance) *10)

    total_score = area_score + shape_score + center_score

    return total_score

def find_best_document(contours, image_center, frame_area):

    best_score = -1
    best_contour = None
    best_ordered = None

    quad_candidates = 0

    for contour in contours[:10]:

        area = cv2.contourArea(contour)

   
        if area < MIN_AREA_RATIO * frame_area:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue


        bx, by, bw, bh = cv2.boundingRect(approx)
        extent = area / float(bw * bh)
        if extent < MIN_EXTENT:
            continue

        quad_candidates += 1

        points = approx.reshape(4, 2)
        ordered = order_points(points)

        width, height = cal_doc_size(ordered)

        if height == 0:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        contour_center = np.array([cx, cy])

        score = cal_score(
            area,
            height,
            width,
            contour_center,
            image_center
        )

        print(f"Area: {area:.0f}  Extent: {extent:.2f}  Score: {score:.1f}")

        if score > best_score:
            best_score = score
            best_contour = approx
            best_ordered = ordered

    print(f"Total contours checked: {len(contours[:10])} | valid rectangle candidates: {quad_candidates}")

    return best_contour, best_ordered, best_score

def warp_document(image, ordered):

    width, height = cal_doc_size(ordered)

    width = int(width)
    height = int(height)

    destination = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        ordered,
        destination
    )

    scanned = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return scanned


def looks_like_paper(scanned):
    """Checks the warped region's actual pixel content -- real paper is
    bright and low in color saturation. Rejects screens, book covers,
    dark objects, or other rectangular things that aren't documents."""

    gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(scanned, cv2.COLOR_BGR2HSV)

    mean_brightness = np.mean(gray)
    mean_saturation = np.mean(hsv[:, :, 1])

    print(f"Brightness: {mean_brightness:.1f}  Saturation: {mean_saturation:.1f}")

    return mean_brightness >= MIN_BRIGHTNESS and mean_saturation <= MAX_SATURATION


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam!")
    exit()

saved = False

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame!")
        break

    frame = cv2.resize(frame, (640, 480))

    original = frame.copy()

    frame_area = frame.shape[0] * frame.shape[1]

    image_center = np.array([
        frame.shape[1] // 2,
        frame.shape[0] // 2
    ])

    processed = preprocess_image(frame)

    # debug view: shows exactly what findContours sees
    cv2.imshow("Debug - Edges", processed)

    contours, hierarchy = cv2.findContours(
        processed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document_contour, ordered, best_score = find_best_document(
        contours,
        image_center,
        frame_area
    )

    if document_contour is not None:

        cv2.drawContours(
            frame,
            [document_contour],
            -1,
            (0, 255, 0),
            2
        )

        for point in ordered:

            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                frame,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

        cv2.imshow("Webcam", frame)

        if best_score >= SCORE_THRESHOLD and not saved:
            scanned = warp_document(original, ordered)

            if looks_like_paper(scanned):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(OUTPUT_DIR, f"scan_{timestamp}.jpg")
                cv2.imwrite(filename, scanned)

                print(f"Document confirmed (score={best_score:.1f}). Saved to {filename}")

                cv2.imshow("Scanned Document", scanned)
                cv2.waitKey(1500)  

                saved = True
                break
            else:
                print("Rectangle detected but doesn't look like paper -- skipping.")

    else:
        cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
