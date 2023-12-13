"""Adapter utils."""

import json
import logging
import os
from abc import abstractmethod
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)  # 获取logger对象


class BaseAdapter(nn.Module):
    """Base adapter.  # 基础适配器

    Can be subclassed to implement custom adapters.  # 可以被子类化以实现自定义适配器
    To implement a custom adapter, subclass this class and implement the
    following methods:  # 要实现自定义适配器，请子类化此类并实现以下方法：
        - get_config_dict  # 获取配置字典
        - forward  # 前向传递

    """

    @abstractmethod  # 抽象方法
    def get_config_dict(self) -> Dict:
        """Get config dict."""  # 获取配置字典

    @abstractmethod
    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass."""  # 前向传递

    def save(self, output_path: str) -> None:
        """Save model."""  # 保存模型
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path: str) -> "BaseAdapter":
        """Load model."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        model = cls(**config)  # **config传入的是字典
        model.load_state_dict(  # 加载模型
            torch.load(  # torch.load()函数用于加载torch.save()函数保存的对象
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model


class LinearLayer(BaseAdapter):
    """Linear transformation.  # 线性变换

    Args:
        in_features (int): Input dimension.  # 输入维度
        out_features (int): Output dimension.  # 输出维度
        bias (bool): Whether to use bias. Defaults to False.  # 是否使用偏差。默认为False。

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features  # 输入维度
        self.out_features = out_features  # 输出维度
        self.bias = bias  # 是否使用偏差
        self.linear = nn.Linear(in_features, out_features, bias=bias)  # 线性变换
        # seed with identity matrix and 0 bias  # 用单位矩阵和0偏差种子
        # only works for square matrices  # 仅适用于方阵
        self.linear.weight.data.copy_(torch.eye(in_features, out_features))
        if bias:
            self.linear.bias.data.copy_(torch.zeros(out_features))

    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass (Wv)."""  # 前向传递（Wv）
        return self.linear(embed)

    def get_config_dict(self) -> Dict:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
        }


def get_activation_function(name: str) -> Callable:
    """Get activation function.  # 获取激活函数

    Args:
        name (str): Name of activation function.  # 激活函数的名称

    """
    activations: Dict[str, Callable] = {
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "leaky_relu": F.leaky_relu,
        # add more activations here as needed  # 根据需要添加更多激活
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[name]


class TwoLayerNN(BaseAdapter):
    """Two-layer transformation.  # 两层变换

    Args:
        in_features (int): Input dimension.  # 输入维度
        hidden_features (int): Hidden dimension.  # 隐藏维度
        out_features (int): Output dimension.  # 输出维度
        bias (bool): Whether to use bias. Defaults to False.  # 是否使用偏差。默认为False。
        activation_fn_str (str): Name of activation function. Defaults to "relu".  # 激活函数的名称。默认为“relu”。

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = False,
        activation_fn_str: str = "relu",
        add_residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.activation_fn_str = activation_fn_str

        self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
        self.linear2 = nn.Linear(hidden_features, out_features, bias=True)
        # self.linear1.weight.data.copy_(torch.zeros(hidden_features, in_features))
        # self.linear2.weight.data.copy_(torch.zeros(out_features, hidden_features))
        # if bias:
        #     self.linear1.bias.data.copy_(torch.zeros(hidden_features))
        #     self.linear2.bias.data.copy_(torch.zeros(out_features))

        self._activation_function = get_activation_function(activation_fn_str)
        self._add_residual = add_residual
        # if add_residual, then add residual_weight (init to 0)
        self.residual_weight = nn.Parameter(torch.zeros(1))

    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass (Wv).  # 前向传递（Wv）

        Args:
            embed (Tensor): Input tensor.  # 输入张量

        """
        output1 = self.linear1(embed)
        output1 = self._activation_function(output1)
        output2 = self.linear2(output1)

        if self._add_residual:
            # print(output2)
            # print(self.residual_weight)
            # print(self.linear2.weight.data)
            output2 = self.residual_weight * output2 + embed

        return output2

    def get_config_dict(self) -> Dict:
        """Get config dict."""  # 获取配置字典
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "activation_fn_str": self.activation_fn_str,
            "add_residual": self._add_residual,
        }
