import numpy as np
import matplotlib.pyplot as plt


r = np.load("prefill_1.npy")

episodes = np.arange(1, len(r) + 1) * 100  # 每个点对应100局


plt.figure()
plt.plot(episodes, r, label="prefill")
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("Episode reward vs Episode")
plt.legend()
plt.grid(True)
plt.show()
