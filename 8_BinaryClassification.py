import torch

# case 1
N = 20
random0 = torch.randn(int(N / 2), 1)
random5 = torch.randn(int(N / 2), 1) + 5
class1_data = torch.hstack([random0, random5])
class2_data = torch.hstack([random5, random0])
class1_label = torch.ones(int(N / 2), 1)
class2_label = torch.zeros(int(N / 2), 1)
X = torch.vstack([class1_data, class2_data])
y = torch.vstack([class1_label, class2_label])

print(X)
print(y)

import matplotlib.pyplot as plt

# plt.plot(class1_data[:, 0], class1_data[:, 1], "o")
# plt.plot(class2_data[:, 0], class2_data[:, 1], "ro")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.grid()
# plt.show()

# 모델 만들기
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # case 1, plain
        self.linear = nn.Sequential(
            nn.Linear(2, 10), nn.Sigmoid(), nn.Linear(10, 1), nn.Sigmoid()
        )  # 10,100,1000,10000으로 바꿔가면서 확인

        # # case 1, very simple
        # self.linear = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        # # 아주 simple 한 것도 오히려 괜춘

        # # case 1, deep
        # self.linear = nn.Sequential( # deep하면 오래걸림
        #     nn.Linear(2, 100),
        #     nn.Sigmoid(),
        #     nn.Linear(100, 100),
        #     nn.Sigmoid(),
        #     nn.Linear(100, 100),
        #     nn.Sigmoid(),
        #     nn.Linear(100, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        x = self.linear(x)
        return x


model = MLP()
print(model)
print(model(torch.randn(5, 2)))

from torch import optim

# 모델 학습시키기
LR = 1e-1
EPOCH = 100
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

loss_history = []
grad_history = []
update_size_history = []

model.train()  # train mode로 전환
for ep in range(EPOCH):
    # inference
    y_hat = model(X)
    # loss
    loss = criterion(y_hat, y)
    # update
    optimizer.zero_grad()  # gradient 누적을 막기 위한 초기화
    loss.backward()  # backpropagation
    optimizer.step()  # weight update
    # print loss
    loss_history += [loss.item()]
    print(f"Epoch: {ep+1}, train loss: {loss.item():.4f}")
    print("-" * 20)
    # 1. 예측 2. loss 구해서 3. 미분하고 4. 업데이트

# ---------------------
# 확인용
print(criterion(y_hat, y))
print(torch.sum(-torch.log(y_hat**y * (1 - y_hat) ** (1 - y))) / N)
print(criterion(torch.tensor([0.0]), torch.tensor([1.0])))
# 100이 loss 최대로 나온다.

# plt.plot(range(1, EPOCH + 1), loss_history, "o")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()


# 모델 테스트하기
x1_test = torch.linspace(-10, 10, 30)  # case 2
x2_test = torch.linspace(-10, 10, 30)  # case 2
X1_test, X2_test = torch.meshgrid(x1_test, x2_test)
X_test = torch.cat([X1_test.unsqueeze(dim=2), X2_test.unsqueeze(dim=2)], dim=2)

model.eval()  # test mode로
with torch.no_grad():
    y_hat = model(X_test)
# 1. dropout 혹은 BN 같은거 사용했다면 train mode와 test mode 동작이 다르므로 eval()로 mode를 바꿔줘야
# 2. grad_fn 계산 <- 메모리가 불필요하게 쓰인다

Y_hat = y_hat.squeeze()


plt.figure(figsize=[10, 9])  # figsize = [가로, 세로]
ax = plt.axes(projection="3d")
ax.view_init(elev=25, azim=-140)
ax.plot_surface(X1_test, X2_test, Y_hat.numpy(), cmap="viridis", alpha=0.2)
plt.plot(class1_data[:, 0], class1_data[:, 1], class1_label.squeeze(), "bo")
plt.plot(class2_data[:, 0], class2_data[:, 1], class2_label.squeeze(), "ro")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
