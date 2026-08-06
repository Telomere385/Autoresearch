import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

r = np.load(DATA_DIR / "prefill" / "prefill_1.npy")

episodes = np.arange(1, len(r) + 1) * 100  # 每个点对应100局


plt.figure()
plt.plot(episodes, r, label="prefill")
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("Episode reward vs Episode")
plt.legend()
plt.grid(True)
plt.show()
