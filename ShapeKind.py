from Shapes import Shapes
from collections import namedtuple
from Colors import Color
import math
import numpy as np

Point = namedtuple("Point", ["x", "y"])


class BoundaryPointShape:
    def __init__(self):
        self.__points: list[Point] = []

    def append(self, point: Point) -> None:
        self.__points.append(point)

    def get_points_number(self) -> int:
        return len(self.__points)

    def get_area(self) -> int:
        p1 = self.__points[0]
        p2 = self.__points[1]
        p3 = self.__points[2]
        area = (p1.x * p2.y + p2.x * p3.y + p3.x * p1.y) - (
            p1.x * p3.y + p2.x * p1.y + p3.x * p2.y
        )
        return area
        # raise Exception("Cannot have less than 3 points. Check the code.")


class ShapeKind:

    height = None
    width = None
    GAP = 5

    def __init__(self, image, thresh, x1, y1, w, h, centroids, area, num) -> None:
        self.__image = image
        self.__shape_thresh = thresh[y1 : y1 + h, x1 : x1 + w]
        self.__x1 = x1
        self.__y1 = y1
        self.__w = w
        self.__h = h
        self.__centroids = centroids
        self.__area = area
        self.__shape_thresh = thresh[y1 : y1 + h, x1 : x1 + w]
        self.__num = num
        self.image = image
        self.w = w
        self.h = h
        self.area = area
        self.num = num
        self.shape_thresh = self.__shape_thresh
        self.centroids = centroids

    def get_shape(self):
        min_dim = min(self.__w, self.__h)
        row_1 = self.__shape_thresh[0, :min_dim]
        row_last = self.__shape_thresh[-1, :min_dim]
        col_1 = self.__shape_thresh[:min_dim, 0]
        col_last = self.__shape_thresh[:min_dim, -1]
        boundaries = np.vstack((row_1, row_last, col_1, col_last))
        rough_boundaries = np.sum(boundaries, axis=1) / np.full((1, 4), 255 * min_dim)
        inlined_boundaries = np.isclose(rough_boundaries, 1.0, atol=0.05)
        if np.allclose(rough_boundaries, 1.0, atol=0.05):
            return Shapes.RECTANGLE
        if np.any(inlined_boundaries):
            return Shapes.TRIANGLE
        if math.isclose(
            self.__w / 2.0 * self.__h / 2.0 * math.pi,
            self.__area,
            abs_tol=self.__area * 0.02,
        ):
            return Shapes.ELLIPSE
        points_num, found_area = self.__points_info()
        if math.isclose(found_area, self.__area, abs_tol=self.__area * 0.05):
            return Shapes.RECTANGLE
        # a = self.shape_thresh[23, :]
        # i, val = self.check_row_test(np.where(a==255)[0])
        merged: list[bool] = []
        for i in range(len(self.__shape_thresh)):
            merged.append(self.check_row(self.__shape_thresh[i, :]))
        # merged = np.apply_along_axis(
        #     self.check_row, axis=1, arr=self.__shape_thresh
        # )
        # map(lambda item: np.where(item == 255), self.__shape_thresh)
        if not np.all(merged):
            return Shapes.STAR

        return Shapes.TRIANGLE

    def get_color(self):
        centered_pixel = self.__image[
            int(self.__centroids[1]), int(self.__centroids[0])
        ]

        b, g, r = centered_pixel
        return Color(b, g, r)

    def check_row_test(self, filled_idxs):
        for i in range(len(filled_idxs) - 1):
            if filled_idxs[i] + 1 != filled_idxs[i + 1]:
                return i, False
        return -1, True

    def check_row(self, row):
        filled_idxs = np.where(row == 255)[0]
        for i in range(len(filled_idxs) - 4):
            if (filled_idxs[i + 1] - filled_idxs[i] + 1) > ShapeKind.GAP:
                return False
        return True

    def __points_info(self):
        boundary_shape = BoundaryPointShape()
        shape_thresh = self.__shape_thresh

        row_1 = shape_thresh[0, :]
        coord_row_1 = np.where(row_1 == 255)[0]
        row_last = shape_thresh[-1, :]
        coord_row_last = np.where(row_last == 255)[0]
        col_1 = shape_thresh[:, 0]
        coord_col_1 = np.where(col_1 == 255)[0]
        col_last = shape_thresh[:, -1]
        coord_col_last = np.where(col_last == 255)[0]
        if len(coord_row_1):
            boundary_shape.append(Point(np.mean(coord_row_1), 0))
        if len(coord_row_last):
            boundary_shape.append(Point(np.mean(coord_row_last), self.__h - 1))
        if len(coord_col_1):
            boundary_shape.append(Point(0, np.mean(coord_col_1)))
        if len(coord_col_last):
            boundary_shape.append(Point(self.__w - 1, np.mean(coord_col_last)))

        return boundary_shape.get_points_number(), boundary_shape.get_area()
