import numpy as np
import matplotlib.pyplot as plt


r1 = np.load("ave_reward_eps_0.01.npy")
r2 = np.load("ave_reward_eps_0.001.npy")
r3 = np.load("ave_reward_eps_0.0001.npy")

episodes1 = np.arange(1, len(r1) + 1) * 100  # 每个点对应100局
episodes2 = np.arange(1, len(r2) + 1) * 100
episodes3 = np.arange(1, len(r3) + 1) * 100


plt.figure()
plt.plot(episodes1, r1, label="eps = 0.01")
plt.plot(episodes2, r2, label="eps = 0.001")
plt.plot(episodes3, r3, label="eps = 0.0001")

plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("Episode reward vs Episode (different exploration rates)")
plt.legend()
plt.grid(True)
plt.show()
