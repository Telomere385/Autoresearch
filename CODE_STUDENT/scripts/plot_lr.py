import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

r1 = np.load(DATA_DIR / "rewards" / "ave_reward_lr_0.1.npy")
r2 = np.load(DATA_DIR / "rewards" / "ave_reward_lr_0.4.npy")
r3 = np.load(DATA_DIR / "rewards" / "ave_reward_lr_0.6.npy")
r4 = np.load(DATA_DIR / "rewards" / "ave_reward_lr_0.9.npy")


# episodes1 = np.arange(1, len(r1) + 1)
# episodes2 = np.arange(1, len(r2) + 1)
# episodes3 = np.arange(1, len(r3) + 1)
episodes1 = np.arange(1, len(r1) + 1) * 100  # 每个点对应100局
episodes2 = np.arange(1, len(r2) + 1) * 100  
episodes3 = np.arange(1, len(r3) + 1) * 100  
episodes4 = np.arange(1, len(r4) + 1) * 100 


plt.figure()

plt.plot(episodes1, r1, label="ni = 0.1")
plt.plot(episodes2, r2, label="ni = 0.4")
plt.plot(episodes3, r3, label="ni = 0.6")
plt.plot(episodes4, r4, label="ni = 0.9")

plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("Episode reward vs Episode (different learning rates)")
plt.legend()
plt.grid(True)

plt.show()
