import numpy as np


Q = np.load('Q_last_lr_0.6.npy')

print("Q dtype:", Q.dtype)
print("Q shape:", Q.shape)

# 1) Global min/max
qmin = np.min(Q)
qmax = np.max(Q)
qmean = np.mean(Q)
qstd = np.std(Q)
print(f"Q global min/max: {qmin:.6g}  {qmax:.6g}")
print(f"Q mean/std:       {qmean:.6g}  {qstd:.6g}")

# 2) Range per action (last dim a=0/1)
qmin_a = Q.min(axis=(0,1,2))   # shape (2,)
qmax_a = Q.max(axis=(0,1,2))   # shape (2,)
print("Q min/max per action a=0,1:")
for a in range(Q.shape[-1]):
    print(f"  a={a}: min={qmin_a[a]:.6g}, max={qmax_a[a]:.6g}")

# 3) (Optional) Range per dx-bin (first dim)
qmin_dx = Q.min(axis=(1,2,3))  # shape (dx_bins,)
qmax_dx = Q.max(axis=(1,2,3))
print("Q min/max per dx-bin (index: min .. max):")
for i in range(Q.shape[0]):
    print(f"  dx_bin={i:2d}: {qmin_dx[i]:.6g} .. {qmax_dx[i]:.6g}")