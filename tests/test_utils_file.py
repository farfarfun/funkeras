"""funkeras.utils.file.read_lines 的正常路径、边界与失败路径测试。

`read_lines` 本身只依赖标准库（os），不需要 tensorflow/opencv-python 等重依赖；
之前因为 `funkeras/utils/__init__.py` eager 导入 `.image`/`.util` 而被迫连带要求
装 cv2，这里直接从 `funkeras.utils.file` 导入以验证惰性加载修复生效。
"""

import pytest

from funkeras.utils.file import read_lines


def test_read_lines_strips_surrounding_whitespace(tmp_path):
    """正常路径：每行首尾空白应被去除，顺序保持不变。"""
    file_path = tmp_path / "classes.txt"
    file_path.write_text(" cat \n dog\n  bird  \n", encoding="utf8")

    assert read_lines(str(file_path)) == ["cat", "dog", "bird"]


def test_read_lines_on_empty_file_returns_empty_list(tmp_path):
    """边界：空文件应返回空列表，而不是报错或返回 [""]。"""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf8")

    assert read_lines(str(file_path)) == []


def test_read_lines_expands_user_home(tmp_path, monkeypatch):
    """边界：路径中的 `~` 应该按 HOME 展开后再读取。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "names.txt").write_text("fox\n", encoding="utf8")

    assert read_lines("~/names.txt") == ["fox"]


def test_read_lines_missing_file_raises_file_not_found_error(tmp_path):
    """失败路径：文件不存在时应原样抛出 FileNotFoundError，而不是吞掉异常。"""
    missing_path = tmp_path / "does_not_exist.txt"

    with pytest.raises(FileNotFoundError):
        read_lines(str(missing_path))
