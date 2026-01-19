import torch
import matplotlib.pyplot as plt

x = torch.tensor([150, 160, 170, 175, 185.0])  # 키
y = torch.tensor([55, 70, 64, 80, 75.0])  # 몸무게
N = len(x)
# plt.plot(x, y, "o")
# plt.show()

# 초깃값 설정
a = 0.45
b = -35
x_plot = torch.linspace(145, 190, 100)
y_plot = a * x_plot + b

plt.plot(x, y, "o")
plt.plot(x_plot, y_plot, "r")


# a,b를 바꿔가면서 Loss 값을 일일히 구해서 가장 작아지게 하는 a,b를 선정

a = 0.5 + torch.linspace(-0.2, 0.2, 100)
b = -30 + torch.linspace(-20, 20, 100)

L = torch.zeros(len(b), len(a))
for i in range(len(b)):
    for j in range(len(a)):
        for n in range(N):
            L[i, j] = L[i, j] + (y[n] - (a[j] * x[n] + b[i])) ** 2
L = L / N  # MSE

# 3d plot
plt.figure(figsize=[10, 9])
ax = plt.axes(projection="3d")
A, B = torch.meshgrid(a, b)
ax.plot_surface(A, B, L)
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlim([0, 1000])

plt.figure()
plt.contour(a, b, L, 30)
plt.xlabel("a")
plt.ylabel("b")
plt.grid()
# plt.show()
# ---------------------------------------------
print(torch.min(L))
a_opt = A[L == torch.min(L)]
b_opt = B[L == torch.min(L)]
print(f"optimal a = {a_opt}")
print(f"optimal b = {b_opt}")

x_plot = torch.linspace(145, 190, 100)
y_plot = a_opt * x_plot + b_opt
plt.plot(x, y, "o")
plt.plot(x_plot, y_plot, "r")
plt.show()
