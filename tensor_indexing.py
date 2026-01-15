import torch

a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
# 인덱싱과 슬라이싱, list할 떄와 동일
print(a[0])
print(a[1])
print(a[-1])
print(a[1:4])
print(a[7:])
print(a[:7])
print(a[:])

# 행렬에 대한 인덱싱과 슬라이싱
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A[0])  # 하나만 쓰면 행에 대한 인덱싱 (리스트 속 리스트 생각)
print(A[-1])
print(A[1:])
print(A[:])
print(A[0][2])
print(A[0, 2])  # 2차원 행렬도 동일한데, 리스트와 달리 이런 것도 됨
B = [[1, 2, 3, 4], [5, 6, 7, 8]]
print(B)
print(B[0][2])
# print(B[0,2]) 리스트는 이게 오류
print(A[1, :])
print(A[1, 0:3:2])  # 0이상 3미만 2간격으로
print(A[:, 2])
print(A[:][2])  # A[:]가 그냥 A와 같다고 생각이 가능


# 3차원 행렬 인덱식
A = torch.tensor(
    [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
    ]
)
print(A)
print(A.shape)
print(A[0, 1, 2])

a = torch.tensor([[1, 2, 3, 4]])  # 대괄호가 하나 늘어나면 왼쪽에 shape 같이 추가된다.
print(a.shape)

print(A)
print(
    A[[0, 1, 1, 0], [0, 1, 2, 1], [3, 3, 2, 1]]
)  # A[0,0,3],A[1,1,3],A[1,2,2],A[0,1,1]을 보겠다라는 뜻


# boolean 인덱싱
a = [1, 2, 3, 4, 5, 3, 3]
print(a == 3)  # 여러개 값 들어있는 리스트랑 3 달랑 하나랑 같냐? 다르다!
A = torch.tensor([[1, 2, 3, 4], [5, 3, 7, 3]])
print(
    A > 3
)  # 리스트와 달리 각 성분에 대해 비교해 줌, 비교해서 같은 크기의 true, false로 이루어진 행렬을 줌
print(A[A > 3])  # True, False가 담긴 행렬로 인덱싱 가능
# True에 해당되는 애들을 인덱싱 해온다

A[A > 3] = 100  # A에서 4,5,7에 해당됨
print(A)

A = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
B = torch.tensor([True, False, False, True])
print(A[B, :])  # 0행, 3행 슬라이싱

b = torch.tensor([1, 2, 3, 4])
print(b[[True, True, False, False]])  # 텐서 아니고 그냥 리스트여도 됨
c = [1, 2, 3, 4]
# print(c[[True,True,False,False]]) 이건 안됨 error

# tensor로 인덱싱
a = torch.tensor([1, 2, 3, 4, 5])
A = a[2]
print(A)
A = a[torch.tensor(2)]  # torch.tensor를 안에다가?
print(A)
A = a[torch.tensor([2, 3, 4])]
print(A)
A = a[torch.tensor([[2, 2, 2], [3, 3, 3]])]
print(A)  # 인덱싱된 애들로 2행 3열짜리 행렬을 만든다

a = torch.tensor([1, 2, 3])
print(a[torch.tensor([[1, 1, 1], [2, 2, 2]])])
# print(a[[1, 1, 1], [2, 2, 2]]) 그냥 리스트 넣으면 에러

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a[torch.tensor(0)])
A = a[torch.tensor([[0, 1], [1, 1]])]  # [a[0], a[1]] [a[1],a[1]]
print(
    A.shape
)  # 예를 들어, a[0] = tensor([1,2,3])과 같이 1차원 텐서이므로 한 차원이 뒤에 늘어나서 2,2,"3"이 된다
print(A)  # Segmentation 결과 그림 보여줄 때 사용!

# segmentation 결과 그림 보여줄 때 사용!
b = torch.tensor(
    [
        [255, 255, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
        [70, 80, 75],
        [0, 0, 4],
        [60, 100, 255],
    ]
)
import matplotlib.pyplot as plt

plt.imshow(b[torch.tensor([[0, 1], [2, 2]])])
plt.show()

# 도전해 보세요!
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
A = a[torch.tensor([[0, 1], [1, 1]])]
# A와 같은 것을 리스트로 인덱싱을 통해 얻으려면?
