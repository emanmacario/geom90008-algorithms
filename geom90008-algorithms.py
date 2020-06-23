"""
Auxiliary functions for formulas found in
GEOM90008 Foundations of Spatial Information

Author: Emmanuel Macario
Date: 23/06/20
Last Modified: 23/06/20
"""
from __future__ import division
import numpy as np


def bilinear_interpolate(x, y, tl, tr, bl, br):
    """
    :param x: width
    :param y: height
    :param tl: v01 value
    :param tr: v11 value
    :param bl: v00 value
    :param br: v10 value
    :return: bilinear interpolation
    """
    a = (1-x)*(1-y) * bl
    b = x*(1-y) * br
    c = (1-x)*y * tl
    d = x*y * tr
    return a + b + c + d


def manhattan_distance(i, j):
    """
    :param i: point one
    :param j: point two
    :return: manhattan distance
    """
    xi, yi = i
    xj, yj = j
    return np.abs(xi-xj) + np.abs(yi-yj)


# CHECKED
def euclidean_distance(a, b):
    """
    :param a: point one
    :param b: point two
    :return: Euclidean distance
    """
    return np.linalg.norm(a-b)


# CHECKED
def polygon_centroid(points):
    """
    :param points: list of points
    :return: centroid
    """
    n = len(points)
    x_total = 0
    y_total = 0
    for p in points:
        x, y = p
        x_total += x
        y_total += y
    print(f"Centroid: ({x_total/n}, {y_total/n})")


# CHECKED
def polygon_area(points):
    """
    :param points: list of points counter-clockwise (e.g. [(a, b), ..., (c, d)])
    :return: area
    """
    n = len(points)
    area = 0
    for i in range(n):
        product = (points[i%n][1] + points[(i+1)%n][1]) * (points[i%n][0] - points[(i+1)%n][0])
        print(product)
        area += product

    area /= 2
    print(f"Area: {area}")


def point_relative_to_line(p, q, r):
    """
    :param p: line point one (numpy array)
    :param q: line point two (numpy array)
    :param r: point (numpy array)
    :return: direction of r relative to line pq
    """
    d = det(p, q, r)
    print(f"Det(p,q,r) = {d}")
    if d > 0:
        print(f"r: {r} is LEFT of line pq: ({p}, {q})")
    elif d < 0:
        print(f"r: {r} is RIGHT of line pq: ({p}, {q})")
    else:
        print(f"r: {r} is LINEARLY DEPENDENT on line pq: ({p}, {q})")


def line_intersection(p1, p2, p3, p4):
    """
    :param p1: start l1 (list or np array)
    :param p2: end l1 (list or np array)
    :param p3: start l2 (list or np array)
    :param p4: end l2 (list or np array)
    :return: True if lines intersect else False
    """
    a = det(p1, p3, p4)
    b = det(p2, p3, p4)
    c = det(p3, p1, p2)
    d = det(p4, p1, p2)

    print(f"det(p1, p3, p4) = {a}")
    print(f"det(p2, p3, p4) = {b}")
    print(f"det(p3, p1, p2) = {c}")
    print(f"det(p4, p1, p2) = {d}")

    intersect = a * b < 0 and c * d < 0

    if intersect:
        print("Lines l1 and l2 INTERSECT")
    else:
        print("Lines l1 and l2 DO NOT INTERSECT")


def intersection_point(p1, p2, p3, p4):
    """
    :param p1: start l1
    :param p2: end l1
    :param p3: start l2
    :param p4: end l2
    :return: point of intersection (if exists)
    """

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False

    L1 = line(p1, p2)
    L2 = line(p3, p4)
    R = intersection(L1, L2)
    if R:
        print(f"Intersection detected: {R}")
    else:
        print("No single intersection point detected")


##################
# HELPER FUNCTIONS
##################


def det(p, q, r):
    """
    :param p: point (list or np array)
    :param q: point (list or np array)
    :param r: point (list or np array)
    :return: determinant
    """
    X = np.array([[1] + p, [1] + q, [1] + r])
    return np.linalg.det(X)


def main():
    """
    a = np.array([0, 0])
    b = np.array([1, 1])
    answer = euclidean_distance(a, b)
    answer = manhattan_distance(a, b)
    print(answer)
    """

    points = [(14, 69), (67, 9), (91, 29), (68, 61), (73, 73), (106, 96), (92, 116)]
    polygon_area(points)

    p = [10, 10]
    q = [40, 40]
    r = [40, 20]
    point_relative_to_line(p, q, r)

    p1 = [10, 10]
    p2 = [40, 40]
    p3 = [40, 20]
    p4 = [20, 30]
    line_intersection(p1, p2, p3, p4)

    intersection_point(p1, p2, p3, p4)


if __name__ == "__main__":
    main()
