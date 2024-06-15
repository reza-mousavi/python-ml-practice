from math import fabs, sqrt, pow
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
Feature = namedtuple('Feature', ['a', 'b'])


def actual_y(x, feature: Feature):
    return x * feature.a + feature.b


def distance(point: Point, feature: Feature):
    nominator = fabs(feature.a * point.x + feature.b - point.y)
    denominator = sqrt(pow(feature.a, 2) + 1)
    return 0 if nominator == 0 else nominator / denominator


def cost(points: list[Point], feature: Feature):
    res = 0.0
    for pt in points:
        act = actual_y(pt.x, feature)
        res += pow(act - pt.y, 2)
    m = len(points)
    res = res / (2 * m)
    return res



