from typing import Callable, Iterable, Union, Any

import numpy as np
import torch

Matrix = Union[np.ndarray, torch.Tensor]


class SlidingWindow2d:
    def __init__(
        self,
        size: Union[tuple[int, int], int] = (1, 1),
        stride: Union[tuple[int, int], int] = 1,
    ):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, tuple) and len(size) == 1:
            size = (size[0], size[0])
        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, tuple) and len(size) == 1:
            size = (stride[0], stride[0])
        self.size = size

        def func(y: Matrix) -> Matrix:
            return y

        self.func = func
        self.stride = stride
        self.mat: Union[None, Matrix] = None

    def set_mat(self, mat: Matrix):
        if mat is None:
            raise ValueError("mat should be a Matrix")
        self.mat = mat

    def _slide(self, slide_mat: Matrix):
        size_0 = self.size[0]
        size_1 = self.size[1]
        slide_res: list[list[Matrix]] = []
        for i in range(0, slide_mat.shape[0] - size_0 + 1):
            slide_res.append([])
            for j in range(0, slide_mat.shape[1] - size_1 + 1):
                curr_slide = slide_mat[i : i + self.size[0], j : j + self.size[1]]
                curr_res = self.func(curr_slide)
                slide_res[i].append(curr_res)
        return torch.tensor(slide_res)

    def __call__(self, mat: Matrix) -> Matrix:
        if type(mat) not in (np.ndarray, torch.Tensor):
            raise ValueError("Input should be Matrix type")
        if len(mat.shape) not in (2, 3, 4):
            raise ValueError("Matrix must be 2d, 3d or 4d")
        return self._slide(mat)


class MaxPool2d(SlidingWindow2d):
    def __init__(
        self,
        size: Union[tuple[int, int], int] = (1, 1),
        stride: Union[tuple[int, int], int] = 1,
        *args,
        **kwargs
    ):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, tuple) and len(size) == 1:
            size = (size[0], size[0])
        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, tuple) and len(size) == 1:
            size = (stride[0], stride[0])
        self.size = size
        super().__init__(*args, **kwargs)

        def get_max(slice: Matrix) -> Matrix:
            return slice.max()

        self.func = get_max


class AvgPool2d(SlidingWindow2d):
    def __init__(
        self,
        size: Union[tuple[int, int], int] = (1, 1),
        stride: Union[tuple[int, int], int] = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, tuple) and len(size) == 1:
            size = (size[0], size[0])
        if isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, tuple) and len(size) == 1:
            size = (stride[0], stride[0])
        self.size = size

        def get_mean(slice: Matrix) -> Matrix:
            return slice.mean()

        self.func = get_mean

    def __call__(self, mat: Matrix) -> Matrix:
        if type(mat) not in (np.ndarray, torch.Tensor):
            raise ValueError("Input should be Matrix type")
        shape = mat.shape
        if len(shape) not in (2, 3, 4):
            raise ValueError("Matrix must be 2d, 3d or 4d")
        if len(shape) == 3:
            means = []
            for channel in mat:
                means.append(self._slide(channel))
            return torch.tensor([means])
        if len(shape) == 2:
            return self._slide(mat)
        return torch.tensor([])


class GlobalAvgPool(SlidingWindow2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, mat: Matrix) -> Matrix:
        if type(mat) not in (np.ndarray, torch.Tensor):
            raise ValueError("Input should be Matrix type")
        if len(mat.shape) not in (2, 3, 4):
            raise ValueError("Matrix must be 2d, 3d or 4d")
        if len(mat.shape) == 2:
            return mat.mean()
        if len(mat.shape) == 3:
            means = []
            for channel in mat:
                means.append([channel.mean()])
            return torch.tensor(means)
        return torch.tensor([])