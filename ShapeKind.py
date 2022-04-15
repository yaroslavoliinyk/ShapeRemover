from Shapes import Shapes
from collections import namedtuple
from Colors import Color
import math
import numpy as np

Point = namedtuple("Point", ["x", "y"])


class AreaCalculator:
    """Obtaining 3 points, we can calculate th area of the shape"""

    def __init__(self):
        self.points: list[Point] = []

    def append(self, point: Point) -> None:
        self.points.append(point)

    def get_area(self) -> int:
        p1 = self.points[0]
        p2 = self.points[1]
        p3 = self.points[2]
        area = (p1.x * p2.y + p2.x * p3.y + p3.x * p1.y) - (
            p1.x * p3.y + p2.x * p1.y + p3.x * p2.y
        )
        return area


class ShapeKind:
    """Detects the shape of the object"""

    height = None
    width = None
    GAP = 5

    def __init__(self, image, thresh, x1, y1, w, h, centroids, area, num):
        self.image = image
        self.w = w
        self.h = h
        self.area = area
        self.num = num
        self.shape_thresh = thresh[y1: y1 + h, x1: x1 + w]
        self.centroids = centroids

    def get_shape(self):
        rough_boundaries, inlined_boundaries = self.get_boundaries()
        if np.allclose(rough_boundaries, 1.0, atol=0.05):
            return Shapes.RECTANGLE
        if np.any(inlined_boundaries):
            return Shapes.TRIANGLE
        if math.isclose(
            self.w / 2.0 * self.h / 2.0 * math.pi,
            self.area,
            abs_tol=self.area * 0.02,
        ):
            return Shapes.ELLIPSE
        found_area = self.find_area()
        if math.isclose(found_area, self.area, abs_tol=self.area * 0.05):
            return Shapes.RECTANGLE
        merged: list[bool] = []
        for i in range(len(self.shape_thresh)):
            merged.append(self.check_star(self.shape_thresh[i, :]))
        if not np.all(merged):
            return Shapes.STAR
        return Shapes.TRIANGLE

    def get_boundaries(self):
        min_dim = min(self.w, self.h)
        row_1 = self.shape_thresh[0, :min_dim]
        row_last = self.shape_thresh[-1, :min_dim]
        col_1 = self.shape_thresh[:min_dim, 0]
        col_last = self.shape_thresh[:min_dim, -1]
        boundaries = np.vstack((row_1, row_last, col_1, col_last))
        rough_boundaries = np.sum(boundaries, axis=1) / np.full((1, 4), 255 * min_dim)
        inlined_boundaries = np.isclose(rough_boundaries, 1.0, atol=0.05)
        return rough_boundaries, inlined_boundaries

    def get_color(self) -> str:
        centered_pixel = self.image[
            int(self.centroids[1]), int(self.centroids[0])
        ]
        b, g, r = centered_pixel
        return str(Color(b, g, r))

    def check_star(self, row):
        filled_idxs = np.where(row == 255)[0]
        for i in range(len(filled_idxs) - 4):
            if (filled_idxs[i + 1] - filled_idxs[i] + 1) > ShapeKind.GAP:
                return False
        return True

    def find_area(self):
        calc = AreaCalculator()
        row_1 = self.shape_thresh[0, :]
        coord_row_1 = np.where(row_1 == 255)[0]
        row_last = self.shape_thresh[-1, :]
        coord_row_last = np.where(row_last == 255)[0]
        col_1 = self.shape_thresh[:, 0]
        coord_col_1 = np.where(col_1 == 255)[0]
        col_last = self.shape_thresh[:, -1]
        coord_col_last = np.where(col_last == 255)[0]
        if len(coord_row_1):
            calc.append(Point(np.mean(coord_row_1), 0))
        if len(coord_row_last):
            calc.append(Point(np.mean(coord_row_last), self.h - 1))
        if len(coord_col_1):
            calc.append(Point(0, np.mean(coord_col_1)))
        if len(coord_col_last):
            calc.append(Point(self.w - 1, np.mean(coord_col_last)))

        return calc.get_area()
