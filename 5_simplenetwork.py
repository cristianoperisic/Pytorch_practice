import torch
from torchviz import make_dot
from torch import nn

x = torch.tensor([1.0])
model = nn.Linear(1, 1)  # 입력 node 한 개, 출력 node 한 개인 layer 만듦
print(model)

print(model.weight)  # 만들면서 initialze 함
print(model.bias)

y = model(x)
print(y)

y2 = x @ model.weight + model.bias  # 식으로 나타내 본다면..
print(y2)

dot = make_dot(y)
# dot.render("graph", format="png", cleanup=True)
# ---------------------------------------------------
print()

fc1 = nn.Linear(1, 3)  # fully-connected
fc2 = nn.Linear(3, 1)

print(fc1.weight)
print(fc1.bias)
print(fc2.weight)
print(fc2.bias)

x = torch.tensor([1.0])
x = fc1(x)
print(x)
y = fc2(x)
print(y)

x = torch.tensor([1.0])
y2 = (
    x @ fc1.weight.T + fc1.bias
) @ fc2.weight.T + fc2.bias  # pytorch는 weight을 transpose 된 채로 가지고 있다
print(y2)

# -------------------------------------------------------
print()

model = nn.Linear(2, 3)
x = torch.randn(2)
print(x)
print(model(x))
# nn.Linear는 데이터의 shape의 마지막 차원이 '채'로(1D data) 들어오길 기대하는 녀석이다
# (입력 노드 하나가 곧 하나의 채널(피쳐) 값을 의미)
# ('채널'은 'TV 채널'의 '채널'같이 특정 유형의 정보를 전달하는 통로. 즉, 피쳐(특징)와 의미적으로 거의 비슷함)

# ----------------------------------------------------------------
print()

model = nn.Linear(
    2, 3
)  # 따라서, 데이터 여러 개를 통과시키고 싶다면 개x채 의 형태로 줘야 함 ('채x개'나 '개*채' 이런 식으로 말고!)
x = torch.randn(5, 2)  # 개x체 ==> 두 개의 채널 값(키, 몸무게)을 가지는 데이터(사람) 5개

print(x)
print(model(x))

x = torch.randn(4, 5, 2)  # nn.Linear는 이거를 개x개x채로 들어왔다고 생각함
print(model(x).shape)

x = torch.rand(2, 3, 6, 4, 5, 2)
print(model(x).shape)

# 그렇다면 왜 웨이트 행렬에 T? weight 도 데이터와 마찬가지로 개x채 형태로 만들기 위함!
# 예를 들어 nn.Linear(2,3) 이면 두 채널 값을 사용하는 세 '개'의 필터를 통과하는 것이라 3x2 가 된다!
# 데이터의 개체는 두 채널 값을 가지는 열 개의 데이터 (10x2)
# 웨이트의 개체는 두 채널 값을 이용하는 세 개의 필터
# --------------------------------------------------------------------
print()

fc1 = nn.Linear(1, 3)
fc2 = nn.Linear(3, 1)

x = torch.tensor([1.0])
x = fc1(x)
print(x)
x = fc2(x)
print(x)

model = nn.Sequential(fc1, fc2)  # layer 풀칠
x = torch.tensor([1.0])
print(model(x))
# ----------------------------------------------------------------------
print()

model = nn.Sequential(
    nn.Linear(2, 5),  # (in 채널, out_채널) 니까 연결되는 부분이 같아지도록
    nn.Linear(5, 10),
    nn.Linear(10, 3),
)

x = torch.randn(5, 2)
print(x)
print(model(x))
# -----------------------------------------------------------------------
print()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, 3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        return x


model = MyModel()
x = torch.randn(5, 2)
y = model(x)  # model.forward(x) (nn.Module의 __call__에서 forward를 통과시킴)
# model.forward(x) 를 실행시키는 것과 같다는 말
print(y)


# ----------------------------------------------------------------------
# 위에를 Sequential 로 표현
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(2, 5),
            nn.Sigmoid(),
            nn.Linear(5, 10),
            nn.Sigmoid(),
            nn.Linear(10, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


# -----------------------------------------------------------------
print()

print(model.parameters())

# 파라미터 수 구하기
num = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(num)
