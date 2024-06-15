import logging
import timeit
import numpy as np

from gradient_descent import GradientDescent, Point, Feature
from gradient_descent_np import GradientDescentNp


class Main:

    def __init__(self):
        # filename='myapp.log'
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

    def main(self):
        self._logger.info("Running main...")
        self._logger.info("Running Gradient Descent...")

        start = timeit.default_timer()
        # Your statements here
        train_data = [Point(-9, -10.3), Point(-7.5, -5), Point(-5, -4.75), Point(3, 2),
                      Point(4, 5), Point(5, 7), Point(6, 8.3), Point(7.5, 8.1)]

        gd = GradientDescent(train_data, epochs=800)  # epochs=10000, learning_rate=0.01
        init_feature = Feature(0, 0)
        result = gd.optimize(init_feature) # Feature(0, -1, 0)
        optimized_feature = result[-1]['optimized_feature']
        logging.info(f"Ordinary Optimized feature {init_feature} to {optimized_feature}")
        stop = timeit.default_timer()
        logging.info(f"Ordinary Calculation time: {stop - start}", )

        xs = np.array([-9, -7.5, -5, 3, 4, 5, 6, 7.5])
        ys = np.array([-10.3, -5, -4.75, 2, 5, 7, 8.3, 8.1])

        start = timeit.default_timer()
        gdnp = GradientDescentNp(xs, ys, epochs=800)  # epochs=10000, learning_rate=0.01
        result = gdnp.optimize(0, 0)
        optimized_feature = result[-1]['optimized_feature']
        logging.info(f"Numpy Optimized feature {init_feature} to {optimized_feature}")
        stop = timeit.default_timer()
        logging.info(f"Numpy Calculation time: {stop - start}", )
        return None


if __name__ == "__main__":
    Main().main()
