from collections import namedtuple
from ShapeKind import ShapeKind
import cv2
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--image", required=False, help="path to input image", default="shapes.png"
)
ap.add_argument("-rm", "--remove", required=False, help="path to output image")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])
shapes = image.copy()
gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
Shape = namedtuple("Shape", ["SKind", "area", "x", "y", "width", "height"])
# cv2.imshow("Thresh", thresh)
# cv2.waitKey(0)
(numLabels, labels, stats, centroids) = output
ShapeKind.height, ShapeKind.width = shapes.shape[:2]
for i, (x1, y1, w, h, area) in enumerate(stats[1:]):
    height, width = shapes.shape[:2]
    # if area > height * width * 0.0001:
    shape_kind = ShapeKind(image, thresh, x1, y1, w, h, centroids[i + 1], area, i + 1)

    name = shape_kind.get_shape()
    if args["remove"] and name in args["remove"]:
        cv2.rectangle(
            shapes, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 255, 255), -1
        )
    else:
        cv2.circle(
            shapes,
            (int(centroids[i + 1][0]), int(centroids[i + 1][1])),
            10,
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            shapes, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 0), 2,
        )
        cv2.putText(
            shapes,
            f"{shape_kind.get_color(), name} #{i+1}",
            (int(centroids[i + 1][0] - 10), int(centroids[i + 1][1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

cv2.imshow("Image", shapes)
cv2.waitKey(0)
