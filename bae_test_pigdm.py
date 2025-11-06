import numpy as np
import torch
import math

pred_original_sample = torch.randn(1, 16, 5, requires_grad=True)


# 조건
prev_action = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5],
                             [0.2, 0.3, 0.4, 0.5, 0.1],
                             [0.3, 0.4, 0.5, 0.1, 0.2],
                             [0.4, 0.5, 0.1, 0.2, 0.3],
                             [0.5, 0.1, 0.2, 0.3, 0.4],
                             [0.6, 0.7, 0.8, 0.9, 1.0],

                             [0.7, 0.8, 0.9, 1.0, 0.6],
                             [0.8, 0.9, 1.0, 0.6, 0.7],
                             [0.9, 1.0, 0.6, 0.7, 0.8],
                             [1.0, 0.6, 0.7, 0.8, 0.9],
                             [0.1, 0.2, 0.3, 0.4, 0.5],
                             [0.2, 0.3, 0.4, 0.5, 0.1],
                             [0.3, 0.4, 0.5, 0.1, 0.2],
                             [0.4, 0.5, 0.1, 0.2, 0.3],
                             [0.5, 0.1, 0.2, 0.3, 0.4],
                             [0.6, 0.7, 0.8, 0.9, 1.0]]])  # shape (1, 16, 5)
assert prev_action.shape[1] == 16, "prev_action should have 16 action steps"
y = torch.zeros_like(prev_action)
y[:, :10] = prev_action[:, 6:]   # 이전 action에서 앞에 6개 자르기


# w = 조건 가중치 (from RTC) (이전꺼에서 남는 action 10개 라고 가정)
w = [1., 9/10, 8/10, 7/10, 6/10, 5/10, 4/10, 3/10, 2/10, 1/10, 0., 0., 0., 0., 0., 0.]
w = w * torch.expm1(w) / (math.e - 1.0)


# grad = 예측한 x_t 에 대해 x_t로 미분한 값 (grad 잘 살려서 가져오기)
error = (y - pred_original_sample) * w[:, None]
# y, origin_sample = (B, T, Da) / w = (T,) -> w[:, None] = (T, 1)

vjp = torch.autograd.grad(outputs=pred_original_sample,
                                  inputs=sample,
                                  grad_outputs=error,
                                  retain_graph=True,
                                  create_graph=False)

        # 계수가 필요없나? 아닌가?
guidance = 0.00001 * vjp