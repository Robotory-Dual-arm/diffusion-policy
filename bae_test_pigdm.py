import torch

# -----------------------------
# 1. 전체 상태 x (4x4 "이미지")
# -----------------------------
x_true = torch.tensor([
    [ 1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.],
    [ 9., 10., 11., 12.],
    [13., 14., 15., 16.]
])

# diffusion 모델의 예측 (랜덤)
x_hat_t = torch.randn_like(x_true)

# -----------------------------
# 2. 선택된 row index (예: 위쪽 절반만 관측)
# -----------------------------
obs_rows = torch.tensor([0, 1])   # 관측되는 행 인덱스 (상단 절반)

# -----------------------------
# 3. Forward operator H(x)
# -----------------------------
def H(x):
    """마스크된 부분만 추출"""
    return x[obs_rows, :]

# -----------------------------
# 4. Pseudoinverse operator H⁺(y)
# -----------------------------
def H_pinv(y):
    """관측값 y를 full image 크기로 되돌림"""
    x_full = torch.zeros_like(x_true)
    x_full[obs_rows, :] = y
    return x_full

# -----------------------------
# 5. Forward measurement
# -----------------------------
y = H(x_true)  # 관측된 부분

# -----------------------------
# 6. ΠGDM 보정항 g 계산
# -----------------------------
g = H_pinv(y) - H_pinv(H(x_hat_t))

# -----------------------------
# 7. 결과 출력
# -----------------------------
print("True x:")
print(x_true)
print("\nObserved y = H(x):")
print(y)
print("\nPredicted x_hat_t:")
print(x_hat_t)
print("\nBackprojected H⁺(y):")
print(H_pinv(y))
print("\nBackprojected H⁺(H(x_hat_t)):")
print(H_pinv(H(x_hat_t)))
print("\nCorrection g = H⁺(y) - H⁺(H(x_hat_t)):")
print(g)
