import matplotlib.pyplot as plt

models = [
    "Random Forest",
    "CNN",
    "CNN + Aug",
    "CNN-LSTM"
]

accuracy = [
    69.94,
    53.76,
    57.23,
    46.05
]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy (%)")

for i, v in enumerate(accuracy):
    plt.text(i, v + 1, str(v))

plt.tight_layout()

plt.savefig("outputs/accuracy_comparison.png")

plt.show()
