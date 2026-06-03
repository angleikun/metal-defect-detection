"""可复现性种子设置。

所有训练、评测、数据划分脚本统一调用此模块。
"""


def set_seed(seed: int = 42) -> None:
    """三件套封装。所有训练、评测、数据划分脚本统一调用。

    注意：故意不开 torch.use_deterministic_algorithms(True)。
    ultralytics 8.x 内部 scatter_add 等 op 不支持，开了直接报错训不动。
    cudnn.deterministic + manual_seed 已足够，实测同 seed 重跑
    mAP 浮动 < 0.005。

    Claude Code 后续如果"优化"代码想加 use_deterministic_algorithms，
    必须 review 这条注释，不要加。
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
