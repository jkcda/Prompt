"""数据表模型。

新增表时在这里导入一次，`SQLModel.metadata` 才能感知到。
"""

from .job import JobRecord, PromptRecord

__all__ = ["JobRecord", "PromptRecord"]
