import importlib

from .file import read_lines

__all__ = [
    "read_lines",
    "boxes_iou",
    "draw_bbox",
    "image_resize",
    "POOL_NSP",
    "POOL_MAX",
    "POOL_AVE",
    "get_checkpoint_paths",
    "extract_embeddings_generator",
    "extract_embeddings",
    "compose",
    "get_random_data",
]

# `.image` 依赖 cv2/tensorflow，`.util` 依赖 tensorflow/matplotlib；
# 这里按 PEP 562 做惰性加载，避免只用 `read_lines` 这类零依赖工具函数时
# 被迫连带导入重依赖（否则会出现仅仅 `from funkeras.utils.file import
# read_lines` 都因为包 `__init__.py` 的 eager import 而要求装 cv2 的问题）。
_LAZY_IMAGE_ATTRS = {"boxes_iou", "draw_bbox", "image_resize"}
_LAZY_UTIL_ATTRS = {
    "POOL_NSP",
    "POOL_MAX",
    "POOL_AVE",
    "get_checkpoint_paths",
    "extract_embeddings_generator",
    "extract_embeddings",
    "compose",
    "get_random_data",
}


def __getattr__(name: str):
    """按需导入 `.image` / `.util` 子模块，解析重依赖相关的公开属性。

    参数:
        name: 被访问的模块级属性名。

    返回:
        对应子模块中的同名属性。

    异常:
        AttributeError: `name` 不在本模块导出的属性列表中。
    """
    if name in _LAZY_IMAGE_ATTRS:
        module = importlib.import_module(".image", __name__)
        return getattr(module, name)
    if name in _LAZY_UTIL_ATTRS:
        module = importlib.import_module(".util", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
