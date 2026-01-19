import torch

a = torch.tensor([1, 2, 3, 4])  # list가 입력으로 들어가는 함수
print(a)
print(type(a))
print(a.dtype)
print(a.shape)
b = torch.tensor([1, 2, 3.1, 5])  # 하나라도 실수면 자동으로 실수 타입
print(b.dtype)
print(b)

A = torch.tensor([[1, 2, 3], [3, 4, 5]])
# A = torch.tensor([[1,2],[3,4,5]]) 이러면 오류남. 행렬이라 행,열의 개수 잘 맞춰줘야..
print(A)
print(A.shape)
print(A.ndim)  # 차원의 수
print(A.numel())  # 전체 성분의 수


print(torch.zeros(5))
print(torch.zeros_like(A))
print(torch.ones(5))
print(torch.zeros(3, 4))
print(torch.arange(3, 10, 2))  # range와 같은데 tensor로 만들어줌
print(torch.arange(0, 1, 0.1))  # 소수점 가능
print(torch.linspace(0, 1, 10))  # 0에서부터 1 포함 10개로

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print(c)

A = torch.tensor([[1, 2, 3], [1, 2, 3]])
B = torch.tensor([[4, 5, 6], [1, 1, 1]])
C = A + B
D = A - B
print(C)
print(D)
print()
print(A * B)  # 곱셈은? 성분끼리의 곱! (Hadamard product)
print(A / B)  # 나누기도 마찬가지
print(B**2)  # 제곱도 각 성분에 대해서 해준다

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[1, 2], [3, 4]])
print(A * B)
print(A @ B)  # 이게 진짜 행렬 곱
