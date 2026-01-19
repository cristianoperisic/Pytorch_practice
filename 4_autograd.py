import torch

x = torch.tensor(
    [1.0],
    requires_grad=True,  # 디폴트 값은 requires_grad=False
)

x = torch.tensor([1.0])
print(x, x.requires_grad)

x.requires_grad = True
print(x, x.requires_grad)

# --------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
y = x**2
print(y)  # PowBackward0 가 붙어있다 !

print(x.grad)
y.backward()  # requires_grad=True인 것에 대해서 미분해줘
print(x.grad)  # y**2을 미분한 2x의 x 값에 1을 대입한 gradient 값

# ---------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
y = x**2
print(y)
# y.retain_grad() # 이걸 하면 y.grad도 볼 수 있다

z = 3 * y
print(z)  # MulBackward0 가 붙어있다!

z.backward()
print(x.grad)  # chain rule로 알아냄
# print(y.grad) y.retain_grad() 했다면 볼 수 있다

# print(y.grad) # warning! 중간건 안된다 requires_grad=False이기 때문
# leaf tensor: requires_grad=True인 tensor
# z는 뿌리 y는 줄기

# --------------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
y = x**2
z = 3 * y

y.backward()  # 이렇게 하면 y에서부터 뒤로 넘어감 (backward!)
print(x.grad)

# ----------------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
a = x**2
b = a + 1
print(b)  # AddBackward0 가 붙어있다!
c = b**2
c.backward()
print(x.grad)

# -----------------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([1.0], requires_grad=True)
z = 2 * x**2 + y**2
print(z)
z.backward()
print(x.grad)
print(y.grad)

# -----------------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([1.0], requires_grad=True)
z = y * x**2
z.backward()
print(x.grad)
print(y.grad)

# -----------------------------------------------------
print()

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.sum(x**2)  # x1**2 + x2**2 + x3**3
y.backward()

print(y)
print(x.grad)  # 스칼라를 벡터로 미분

# ------------------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
x.reqiures_grad = False
# transfer learning 할 때 필요
y = x**2
print(y)
# y.backward()  # error!

# --------------------------------------------------------
print()

x = torch.tensor([1.0], requires_grad=True)
x2 = x.detach()  # detach는 requires_grad=False인 새로운 텐서를 만드는 것.
print(x)
print(x2)
y = x**2
print(y)
y2 = x2**2
print(y2)

# -----------------------------------------------------------
# detach 사용 용도
print()

x = torch.tensor([1.0], requires_grad=True)
y = x**2
z = (
    y.detach()
)  # x로 만든 것을 상수로 사용하고 싶은 것. y.requires_grad=False 뭐 이런 식으로 중간에 상수로 바꿀 수는 없다
w = y + z  # x**2+1과 같다. z는 상수로 쓰이는 느낌임

w.backward()
print(x.grad)

# ------------------------------------------------------------
# 많이 쓰이는 torch.no_grad
print()

x = torch.tensor([1.0], requires_grad=True)
# chain rule을 위해 계속 grad_fn을 update 하니까 grad_fn 잠시 안 계산하고 싶을 때 torch.no_grad
# 모델 테스트 시에는 불필요하게 메모리 쓸 필요가 없기 때문!
with torch.no_grad():
    y = x**2
    print(x.requires_grad)
    print(y)
print(x.requires_grad)
# y.backward() # error!
y = x**2
print(y)

x = torch.tensor([1.0], requires_grad=True)
x.requires_grad = False
y = x**2
print(x.requires_grad)
print(y)
# y.backward() # error! 근데 이건 다시 x.requires_grad = True 바꿔줘야해서 번거롭다
# ---------------------------------------------------------------
from torchviz import make_dot

x = torch.tensor([1.0], requires_grad=True)
# make_dot(x)
dot = make_dot(x**2)


# make_dot(x**2) # (1) 이라고 써있는 것은 shape을 나타냄
# make_dot(x**2+1)
dot = make_dot((x**2 + 1) ** 2)
# dot.render("graph", format="png", cleanup=True)

y = 2 * x
z = 3 + x
r = y + z
dot = make_dot(r)
# dot.render("graph", format="png", cleanup=True)

r.backward()
print(x.grad)
