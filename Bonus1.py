import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 9.9, 100)
p_s1_given_t = 1 / (2 - t / 10)

plt.figure(figsize=(8,5))
plt.plot(t, p_s1_given_t, label="P(S=1 | T > t)", color="blue")
plt.axhline(y=1, linestyle="--", color="gray", label="Certainty (100%)")
plt.title("Probability Luggage Is Still on the Plane Over Time")
plt.xlabel("Time waited (minutes)")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()