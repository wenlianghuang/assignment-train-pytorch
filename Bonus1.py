import matplotlib.pyplot as plt
import numpy as np

# 時間範圍
time = np.linspace(0, 10, 100)

# 條件機率計算
P_S1_given_R0 = np.ones_like(time)  # 始終為 1

# 繪圖
plt.figure(figsize=(8, 5))
plt.plot(time, P_S1_given_R0, label="P(S=1|R=0, t)", color="blue")
plt.xlabel("Time (minutes)")
plt.ylabel("Conditional Probability")
plt.title("Conditional Probability of Luggage Still on Airplane Over Time")
plt.legend()
plt.grid()
plt.show()