import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, mean):
        self.mean = mean
        self.observed_mean = 0
        self.pulls = 0

    def pull(self):
        return np.random.randn() + self.mean

    def update(self, x):
        self.pulls += 1
        self.observed_mean = (1 - 1.0 / self.pulls) * self.observed_mean + 1.0 / self.pulls * x


def run_experiment(bandit_means, explore_chance, pulls):
    bandits = []

    for mean in bandit_means:
        bandits.append(Bandit(mean))

    data = np.empty(pulls)

    for pull in range(pulls):

        # do rng to determine whether to explore or exploit
        p = np.random.random()

        if p < explore_chance:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([bandit.observed_mean for bandit in bandits])

        pull_result = bandits[j].pull()
        bandits[j].update(pull_result)

        # collect pull results for plotting
        data[pull] = pull_result

    cumulative_average = np.cumsum(data) / (np.arange(pulls) + 1)

    plt.plot(cumulative_average)

    for mean in bandit_means:
        plt.plot(np.ones(pulls) * mean)

    plt.xscale('log')
    plt.show()

    for bandit in bandits:
        print(bandit.observed_mean)

    return cumulative_average


def main():
    bandit_means = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    c_1 = run_experiment(bandit_means, 0.1, int(1e5))
    c_05 = run_experiment(bandit_means, 0.05, int(1e5))
    c_01 = run_experiment(bandit_means, 0.01, int(1e5))

    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
