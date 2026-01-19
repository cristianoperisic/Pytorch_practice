import torch
import matplotlib.pyplot as plt

x = torch.tensor([150, 160, 170, 175, 185.0])  # 키
y = torch.tensor([55, 70, 64, 80, 75.0])  # 몸무게
N = len(x)

# 모델 파라미터 초기화
a = torch.tensor([0.45], requires_grad=True)
b = torch.tensor([-35.0], requires_grad=True)

# 하이퍼파라미터 설정
LR = 3e-6
EPOCH = 20
loss_history = []

for ep in range(EPOCH):
    # inference
    y_hat = a * x + b
    # loss
    loss = 0
    for n in range(N):
        loss += (y[n] - (y_hat[n])) ** 2
    loss = loss / N  # MSE
    # UPDATE
    loss.backward()  # backpropagation
    with torch.no_grad():
        a -= LR * a.grad  # weight update
        b -= LR * b.grad  # weight update
    a.grad = torch.tensor([0.0])  # gradient 초기화
    b.grad = torch.tensor([0.0])  # gradient 초기화
    # print loss, item()쓰는 이유는 tensor라서 스칼라 값만 뽑아내려고
    loss_history += [loss.item()]
    print(f"Epoch: {ep+1}, train loss: {loss.item():.4f}")
    # print weight and bias
    print(f"Weight: {a.item():.4f}, Bias: {b.item():.4f}")
    # plot graph
    x_plot = torch.linspace(145, 190, 100)
    y_plot = a.detach() * x_plot + b.detach()
    # plt.figure()
    # plt.plot(x, y, "o")
    # plt.plot(x_plot, y_plot, "r")
    # plt.title(f"Epoch {ep+1}")
    # plt.show()
# ---------------------------------------------------------------
# .grad 초기화 필요한 이유 실험
z = torch.tensor([1.0], requires_grad=True)
for _ in range(2):
    loss = z**2
    loss.backward()
    print(z.grad)  # .grad 초기화 안하면 계속 누적이 됨
    z.grad = torch.tensor([0.0])

# plt.plot(range(1, EPOCH + 1), loss_history)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()


# -------------------------------------------------------------
# 이것이 정석
from torch import nn, optim

x = x.reshape(-1, 1)  # 개채 형태로 바꿔줌
model = nn.Linear(1, 1)
model.weight.data = torch.tensor([[0.45]])
model.bias.data = torch.tensor([-35.0])

LR = 3e-6
EPOCH = 20
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.MSELoss()

loss_history = []
for ep in range(EPOCH):
    # inference
    y_hat = model(x)
    # loss
    loss = criterion(y_hat, y)
    # update
    optimizer.zero_grad()  # gradient 누적을 막기 위한 초기화
    loss.backward()  # backpropagation
    optimizer.step()  # weight update
    # print loss
    loss_history += [loss.item()]
    print(f"Epoch: {ep+1}, train loss: {loss.item():.4f}")
    # print weight and bias
    print(f"Weight: {model.weight.data.item():.4f}, Bias: {model.bias.data.item():.4f}")
    # plot graph
    x_plot = torch.linspace(145, 190, 100)
    y_plot = model.weight.squeeze().detach() * x_plot + model.bias.detach()
    plt.figure()
    plt.plot(x, y, "o")
    plt.plot(x_plot, y_plot, "r")
    plt.title(f"Epoch {ep+1}")
    plt.show()
