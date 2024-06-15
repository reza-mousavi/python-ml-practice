import numpy as np


class GradientDescentNp:
    """
    Calculates gradient descent using numpy array
    """
    def __init__(self,
                 _training_xs: np.ndarray,
                 _training_ys: np.ndarray, epochs=10000, learning_rate=0.01):
        self._training_xs = _training_xs
        self._training_ys = _training_ys
        self._training_size = len(self._training_ys)
        self._epochs = epochs
        self._learning_rate = learning_rate

    def _gradient(self, a_in, b_in):
        """
        Calculates gradient for the training data toward given feature

        :param feature:
        :return:
        """

        guess_y = self._training_xs * a_in + b_in
        ga = ((guess_y - self._training_ys) * self._training_xs).sum()
        gb = (guess_y - self._training_ys).sum()

        ga = ga / self._training_size
        gb = gb / self._training_size

        return ga, gb

    def optimize(self, a=0, b=0) -> list[dict]:
        """
        Run Gradient Descent algorithm on the given dataset

        :param feature:
        :param a:
        :param b:
        :return:
        """
        # Clone object?
        result = []

        for i in range(self._epochs):
            # Calculate the gradient and update the parameters using gradient_function
            cost_from_optimized_feature = self._cost(a, b)
            gradients = self._gradient(a, b)
            result.append({
                'optimized_feature': (a, b),
                'gradient': gradients,
                'cost': cost_from_optimized_feature})

            b = b - (self._learning_rate * gradients[1])
            a = a - (self._learning_rate * gradients[0])

        return result

    def _cost(self, a_in, b_in):
        """
        Calculates training dataset distance from the given feature(line)

        :param feature:
        :return:
        """
        guess = self._training_xs * a_in + b_in
        errys_sq = (guess - self._training_xs) ** 2
        err_sum = errys_sq.sum() / (2 * self._training_size)
        return err_sum
