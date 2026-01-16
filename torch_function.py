import torch

A = torch.randn(3, 3)  # Normal 의 n
B = torch.rand(3, 3)  # 이건 uniform

# 파이토치는 엄격함. 함수 사용할 때 입력도 tensor형태여야하며 아니면 error 뜬다
A = torch.randn(3, 3)
print(A)
print(torch.abs(A))  # 절댓값
print(torch.sqrt(torch.abs(A)))  # 루트
print(torch.exp(A))  # e^A값들
print(torch.log(torch.abs(A)))  # 자연log
print(torch.log(torch.exp(torch.abs(A))))  # torch.abs(A)와 같음
print(torch.log10(torch.tensor(10)))  # 밑이 10인 로그
print(torch.log2(torch.tensor(2)))  # 밑이 2인 로그
print(torch.round(A))  # 반올림
print(torch.round(A, decimals=2))  # 소숫점 2자리까지
print(torch.floor(A))  # 내림
print(torch.ceil(A))  # 올림

# type(torch.pi) -> 이게 float형임
print(torch.sin(torch.tensor(torch.pi / 6)))
print(torch.cos(torch.tensor(torch.pi / 3)))
print(torch.tan(torch.tensor(torch.pi / 4)))
print(torch.tanh(torch.tensor(-10)))
print(type(torch.tensor(1) / 6))

# torch.nan # not a number
print(torch.log(torch.tensor(-1)))  # tensor(nan)이 출력
print(torch.isnan(torch.tensor([1, 2, torch.nan, 3, 4])))
print(torch.isinf(torch.tensor([1, 2, 3, 4, torch.inf])))  # infinite

A = torch.randn(3, 4)
print(A)
print(torch.max(A))  # 행렬 전체에서 최댓값
print(torch.max(A, dim=0))  # 차원에서 0번째
print(torch.max(A, dim=1))  # 차원에서 1번째 , 1D tensor로 바꿔버림
print(torch.max(A, dim=0, keepdims=True))  # 1행 4열 짜리 2D tensor
print(torch.max(A, dim=1, keepdims=True))  # 3행 1열 짜리 2D tensor
print(torch.min(A))
print(torch.min(A, dim=0))
print(torch.min(A, dim=1))
print(torch.argmax(A))  # 가장 큰 값을 가지는 인덱스
print(torch.argmax(A, dim=0))  # 각 열에서 가장 큰 해가 존재하는 인덱스
print(torch.argmax(A, dim=1))  # 각 열에서 가장 큰 해가 존재하는 인덱스

a = torch.randn(6, 1)
print(a)
a_sorted = torch.sort(a, dim=0)
print(a_sorted)

a = torch.randn(6, 1)
print(a)
print(a.sort(dim=0))  # 기본이 오름차순
print(a.sort(dim=0, descending=True))  # 내림차순 설정
print(a)  # a 자체가 정렬되는건 아님

print(torch.max(a))
print(a.max())
print(torch.abs(a))
print(a.abs())


A = torch.rand(3, 4)
print(A)
print(torch.sum(A))
print(torch.sum(A, dim=1))
print(torch.sum(A, dim=1, keepdim=True))
print(torch.mean(A))
print(torch.mean(A, dim=1))
print(torch.mean(A, dim=1, keepdim=True))
print(torch.std(A))

print(A.sum(dim=1, keepdim=True))
print(A.mean(dim=1, keepdim=True))
print(A.std())


A = torch.randint(
    1, 5, size=(12,)
)  # 1부터 5 미만 12개 정수 (1 차원은 (N,)과 같이 표현)
print(A)
print(A.shape)

B = A.reshape(2, 2, 3)
print(B)
print(B.ndim)  # 3차원 행렬이다

# -1은 알아서 채워달라는 뜻
A = torch.arange(
    20,
)
print(A)
print(A.reshape(4, 5))
print(A.reshape(4, -1).shape)  # 4개 행이 되도록 열의 수를 맞춰준다
print(A.reshape(2, 5, -1).shape)
print(A.reshape(2, -1, 5).shape)
print(A.reshape(1, -1).shape)  # 2차원 행 벡터
print(A.reshape(-1, 1).shape)  # 2차원 열 벡터


a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 1])
print(torch.sum(a * b))  # 내적

a = a.reshape(3, 1)
b = b.reshape(3, 1)
print(a.transpose(1, 0) @ b)  # 1번째 차원, 0번째 차원 뒤집어라
print(a.permute(1, 0) @ b)  # permute(0번째에 갖다놓고 싶은거, 1번째에 갖다놓고 싶은거)
print(a.T @ b)
print(a.t() @ b)

A = torch.randn(4, 3, 6)
print(A.permute(0, 2, 1).shape)
print(A.transpose(2, 1).shape)

# ... 활용
x = torch.randn(2, 3, 4, 5, 6)
print(x[1, 2, :, :, :].shape)
print(x[1, 2, ...].shape)
print(x[:, :, :, :, 3].shape)
print(x[..., 3].shape)
print(x[1, :, :, 3, :].shape)
print(x[1, ..., 3, :].shape)


# cat
A = torch.ones(3, 4)
B = torch.zeros(3, 4)
C = torch.vstack((A, B))  # v는 0번째 차원, h는 1번째 차원에 쌓는다
D = torch.hstack((A, B))
E = torch.cat((A, B), dim=0)
F = torch.cat((A, B), dim=1)  # cat 쓰는게 좋음

print(C)
print(D)
print(E)
print(F)

#  squeeze() 딱히 필요없는 1짜리 차원을 없애줌
A = torch.randn(1, 1, 1, 3, 1, 1, 1, 4, 1, 1, 1)
# print(A)
print(A.shape)
print(A.squeeze().shape)
print(A.squeeze(dim=(0, 2, 4, 5)).shape)  # 0,2,4,5번째에 있는 1차원 없앰

A = torch.randn(3, 4)
print(A.unsqueeze(dim=0).shape)
print(A.unsqueeze(dim=1).shape)
print(A.unsqueeze(dim=2).shape)
# 아래처럼 reshape도 가능은 함
print(A.reshape(1, 3, 4).shape)

A = torch.ones(3, 4)
B = torch.zeros(3, 4)
A = A.unsqueeze(dim=0)
B = B.unsqueeze(dim=0)  # 1,3,4차원
C = torch.cat((A, B), dim=0)  # 0번째를 cat했으니 2,3,4차원으로 됨
print(C)
print(C.shape)

A = torch.ones(3, 4)
B = torch.zeros(3, 4)
C = torch.stack((A, B))  # stack은 완전히 차원 같아야함.
print(C)  # unsqueeze 안해도 되는데 대신 완전히 일치해야함

A = torch.tensor([[1, 2], [3, 4]])
B = A
B[0, 0] = 100
print(B)
print(A)  # B의 수정이 A에도 반영된다

# 다르게 하려면
A = torch.tensor([[1, 2], [3, 4]])
B = A.clone()
B[0, 0] = 100
print(B)
print(A)

# @에 대해 좀만 더
A = torch.randn(5, 7)
B = torch.rand(7, 10)
C = A @ B
print(C.shape)

A = torch.randn(32, 5, 7)
B = torch.randn(32, 7, 10)
C = A @ B
print(C.shape)

# 3차원에서 차원 안 맞으면 복제해서 함
A = torch.randn(32, 5, 7)
B = torch.randn(7, 10)
C = A @ B
print(C.shape)  # 32,5,10으로 B를 32개 복제했다

# numpy torch 왔다갔다 가능
import numpy as np

a = np.array([1, 2, 3])
b = torch.tensor([1, 2, 3])
A = torch.tensor(a)
B = b.numpy()
print(type(A))
print(type(B))
