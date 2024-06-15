import logging
import math
from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'])
Feature = namedtuple('Feature', ['a', 'b'])

class GradientDescent:

    _base_feature = Feature(0, 0)

    def __init__(self, training_data: list[Point], epochs=10000, learning_rate=0.01):
        self._training_data = training_data
        self._training_size = len(self._training_data)
        self._epochs = epochs
        self._learning_rate = learning_rate

    def _gradient(self, feature: Feature):
        """
        Calculates gradient for the training data toward given feature

        :param feature:
        :return:
        """
        # Number of training examples
        ff = self._base_feature

        for pt in self._training_data:
            act_y = self._actual_y(pt.x, feature)
            diff = act_y - pt.y
            ff = Feature(a=ff.a + diff * pt.x, b=ff.b + diff)

        return Feature(ff.a/self._training_size, ff.b/self._training_size)

    def optimize(self, feature: Feature = Feature(0, 0)) -> list[dict]:
        """
        Run Gradient Descent algorithm on the given dataset

        :param feature:
        :return:
        """
        # Clone object?
        optimized_feature = Feature(feature.a, feature.b)
        result = []
        for i in range(self._epochs):
            # Cost is calculated to ensure that the calculation diverges!
            cost_from_optimized_feature = self._cost(optimized_feature)
            gradient = self._gradient(optimized_feature)
            result.append({
                'optimized_feature': optimized_feature,
                'gradient': gradient,
                'cost': cost_from_optimized_feature})
            optimized_feature = self._adjust(optimized_feature, gradient)

            if i % math.ceil(self._epochs / 10) == 0:
                logging.info(f"Iter {i:4}: Gradient : {gradient}, "
                             f"cost : {cost_from_optimized_feature}, feature: {optimized_feature}")

        return result

    def _actual_y(self, x, feature: Feature):
        """
        Calculates according y for given x in the feature(line)
        :param x:
        :param feature:
        :return:
        """
        return x * feature.a + feature.b

    def _cost(self, feature: Feature):
        """
        Calculates training dataset distance from the given feature(line)

        :param feature:
        :return:
        """
        res = 0.0
        for pt in self._training_data:
            act = self._actual_y(pt.x, feature)
            res += pow(act - pt.y, 2)
        return res / (2 * self._training_size)

    def _adjust(self, feature, gradient):
        """
        Adjusts the feature(line) by learning_rate
        :param feature:
        :param gradient:
        :return:
        """
        return Feature(a=feature.a - gradient.a * self._learning_rate,
                    b=feature.b - gradient.b * self._learning_rate)
