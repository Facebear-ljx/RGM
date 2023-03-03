import matplotlib.pyplot as plt
import numpy as np


def main():
    data = np.genfromtxt('data/Antmaze_medium_expert_s_l.csv', dtype=float, delimiter=',')
    x, y = data[:, 0], data[:, 1]
    plt.scatter(x, y)
    plt.show()

if __name__ == '__main__':
    main()