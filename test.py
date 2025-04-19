import matplotlib.pyplot as plt
import numpy as np

# 定義各個點的座標
curr = np.array([0, 0])
final = np.array([100, 100])
velocity = np.array([7, 0])
target = final - velocity  # 調整後的目標 (預測點)

# 設置圖表
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-10, 120)
ax.set_ylim(-10, 120)
ax.grid(True)

# 畫出各個點
ax.plot(curr[0], curr[1], 'bo', label='curr (0,0)')
ax.plot(final[0], final[1], 'go', label='final (100,100)')
ax.plot(target[0], target[1], 'ro', label='target (final - velocity)')

# 畫出從 curr 到 final 的箭頭 (直接目標方向)
ax.arrow(curr[0], curr[1], final[0]-curr[0], final[1]-curr[1],
         head_width=3, head_length=5, fc='green', ec='green', linestyle='--')

# 畫出從 curr 到 target 的箭頭 (修正後的目標方向)
ax.arrow(curr[0], curr[1], target[0]-curr[0], target[1]-curr[1],
         head_width=3, head_length=5, fc='red', ec='red')

# 畫出 velocity 向量 (當前速度)
ax.arrow(curr[0], curr[1], velocity[0], velocity[1],
         head_width=3, head_length=5, fc='blue', ec='blue')

# 添加圖例與標籤
ax.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('角色轉向預測：修正前後的目標')

plt.show()
