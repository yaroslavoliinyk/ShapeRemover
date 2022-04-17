# python ShapeDetector.py -m hands_on
from ShapeKind import ShapeKind, Point
import cv2
import argparse
import imutils


class ShapeDetector:
    def __init__(self, imagePath, removedShapes, method):
        self.image = cv2.imread(imagePath)
        self.remove = removedShapes
        if method == "approx_poly":
            img = self.approx_poly()
        elif method == "hands_on":
            img = self.hands_on()
        else:
            raise Exception("Choose correct method: 'approx_poly' or 'hands_on'")
        cv2.imshow("Image", img)
        cv2.waitKey(0)

    def preprocess(self, shapes):
        gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
        return thresh

    def hands_on(self):
        thresh = self.preprocess(self.image.copy())
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        ShapeKind.height, ShapeKind.width = self.image.shape[:2]
        displayable = 1
        for i, (x1, y1, w, h, area) in enumerate(stats[1:]):
            shape_kind = ShapeKind(
                self.image, thresh, x1, y1, w, h, centroids[i + 1], area, i + 1
            )
            name = shape_kind.get_shape()
            if self.remove and name in self.remove:
                cv2.rectangle(
                    self.image,
                    (int(x1), int(y1)),
                    (int(x1 + w), int(y1 + h)),
                    (255, 255, 255),
                    -1,
                )
            else:
                color = shape_kind.get_color()
                cv2.circle(
                    self.image,
                    (int(centroids[i + 1][0]), int(centroids[i + 1][1])),
                    10,
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    self.image,
                    (int(x1), int(y1)),
                    (int(x1 + w), int(y1 + h)),
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    self.image,
                    f"{color, name} #{displayable}",
                    (int(centroids[i + 1][0] - 10), int(centroids[i + 1][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                displayable += 1
        return self.image

    def approx_poly(self):
        resized = self.image.copy()
        ratio = 1
        # # added by myself to be consistent with the other methods
        # self.image = resized.copy()
        # find contours in the thresholded image and initialize the
        # shape detector
        thresh = self.preprocess(resized)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # this function unpacks the contours from the 2 elems tuple
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for i, c in enumerate(cnts):
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            shape = self.detect(c)
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(self.image, [c], -1, (0, 255, 0), 2)
            cv2.putText(
                self.image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2,
            )
        return self.image

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # (x, y, w, h) = cv2.boundingRect(approx)
            p1 = Point(approx[0][0][0], approx[0][0][1])
            p2 = Point(approx[1][0][0], approx[1][0][1])
            p3 = Point(approx[2][0][0], approx[2][0][1])
            w = p1.distance(p2)
            h = p2.distance(p3)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 10:
            shape = "star"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--image", required=False, help="path to input image", default="shapes2.png"
)
ap.add_argument("-rm", "--remove", required=False, help="path to output image")
ap.add_argument(
    "-m",
    "--method",
    required=False,
    default="hands",
    help="User can choose to implement search hands-on or using approxPolyDP",
)
args = vars(ap.parse_args())

sd = ShapeDetector(args["image"], args["remove"], args["method"])
cv2.destroyAllWindows()
