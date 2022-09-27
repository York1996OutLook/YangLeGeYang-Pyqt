"""
常用工具
"""


from typing import Tuple


def get_center_crop_pos(origin_shape: Tuple[int, int] = (720, 1280), cropped_shape: Tuple[int, int] = (100, 200), ) -> Tuple[int, int, int, int]:
    """

    Args:
        origin_shape:
        cropped_shape:

    Returns: Tuple

    """
    start_row = (origin_shape[0] - cropped_shape[0]) // 2
    start_col = (origin_shape[1] - cropped_shape[1]) // 2

    end_row = start_row + cropped_shape[0]
    end_col = start_col + cropped_shape[1]
    return start_row, end_row, start_col, end_col
