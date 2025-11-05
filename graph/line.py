import matplotlib.pyplot as plt

# 数据
steps = [10, 20, 30, 40, 50, 60]
training_loss = [1.376900, 0.851100, 0.801500, 0.727400, 0.722000, 0.728400]

# 绘制折线图
plt.figure(figsize=(8, 5))
plt.plot(steps, training_loss, marker='o', linewidth=2)
plt.title("Training Loss vs Step")
plt.xlabel("Step")
plt.ylabel("Training Loss")
plt.grid(True)

# 保存为 PNG 文件
plt.savefig("training_loss.png", dpi=300)
plt.close()
