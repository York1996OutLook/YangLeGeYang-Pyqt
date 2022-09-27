from collections import defaultdict
import sys
from typing import List, Dict

import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QFrame, QLabel, QMessageBox
from PyQt5.QtCore import pyqtSignal, pyqtBoundSignal

from read_resource import read_stuffs, Stuff

CUBE_WIDTH = 120
CUBE_HEIGHT = 134

np.random.seed(1)


class Cube:
    """
    方块类
    """

    def __init__(self, x: int, y: int, category: str, stuffs_dict: Dict[str, Stuff]):
        self.x: int = x
        self.y: int = y
        self.w: int = CUBE_WIDTH
        self.h: int = CUBE_HEIGHT

        self.tops: List[Cube] = []

        self.bottoms: List[Cube] = []

        self.category: str = category  # 类别名称
        self.stuff: Stuff = stuffs_dict[category]  # 可视化用到的图片


class PicBox(QLabel):
    """
    图片框类，为了实现在主窗体中获得点击label的消息，并且可以将鼠标位置传递给主窗体；
    """
    click_finished: pyqtBoundSignal = pyqtSignal(QtGui.QMouseEvent)

    def __init__(self):
        super(PicBox, self).__init__()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        """
        释放鼠标的时候，触发点击事件
        :param ev:
        :return:
        """
        self.click_finished.emit(ev)


class Game:
    """
    yang le ge yang游戏类
    """

    def __init__(self):
        self.cube_total_count = 20 * 3  # 一共多少个方块

        self.max_move_up_cube_nums = 3  # 点击按钮最多会有多少方块移动到空白区域

        self.width = CUBE_WIDTH * 8
        self.height = CUBE_HEIGHT * 12

        self.main_area_height = CUBE_HEIGHT * 8
        self.slots_area_height = CUBE_HEIGHT * 2
        self.move_up_area_height = CUBE_HEIGHT * 2

        self.cube_start_x_range = (CUBE_WIDTH, self.width - 2 * CUBE_WIDTH)  # 左右各空一格
        self.cube_start_y_range = (CUBE_HEIGHT, self.height - 2 * CUBE_HEIGHT - self.move_up_area_height - self.slots_area_height)

        self.cover_iou_thresh = 0.05  # 两个方块的遮挡阈值大于该值，认为是遮挡

        self.cubes: List[Cube] = []  # 除了slots位置的所有方块

        self.stuffs = read_stuffs()  # 读取素材
        self.cls_count = 10  # 最多出现多少种类别的方块

        self.slot_cubes: List[Cube] = []  # 槽中的方块
        self.max_slots_cubes = 7
        self.slots_positions = self.get_slots_positions()  # x,y
        self.move_up_positions = self.get_init_move_up_positions()
        self.move_up_cubes_dict = defaultdict(list)  # 1,2,3
        self.move_up_cubes_tile_offset_x = 10
        self.move_up_cubes_tile_offset_y = 0

        self.top_cubes = []

    def get_init_move_up_positions(self):
        """
        获得初始化move up的位置，如果有新的，则偏移10然后进行覆盖；
        :return:
        """
        left_right_interval = 200
        move_up_cubes_interval = (self.width - self.max_move_up_cube_nums * CUBE_WIDTH - left_right_interval * 2) // (self.max_move_up_cube_nums - 1)

        start_y = self.main_area_height

        positions = []
        for i in range(self.max_move_up_cube_nums):
            positions.append((left_right_interval + (move_up_cubes_interval + CUBE_WIDTH) * i, start_y))

        return positions

    def get_slots_positions(self):
        """
        获取底部方块的位置们
        :return:
        """
        slots_positions = []

        slots_start_x = 10
        slots_interval = 10
        slots_start_y = self.main_area_height + self.move_up_area_height
        for i in range(self.max_slots_cubes):
            start_x = slots_start_x + i * (CUBE_WIDTH + slots_interval)
            slots_positions.append((start_x, slots_start_y))
        return slots_positions

    def random_cubes(self):
        """
        随机生成cube
        刚开始的时候，cube的范围会较为居中，意味着游戏难度大，由progress控制
        :return:
        """
        all_cats = list(self.stuffs.keys())
        np.random.shuffle(all_cats)
        cur_cats = list(np.random.choice(all_cats[:self.cls_count], self.cube_total_count // 3))
        cur_cats *= 3
        np.random.shuffle(cur_cats)

        center_x = sum(self.cube_start_x_range) // 2
        center_y = sum(self.cube_start_y_range) // 2
        x_scope = self.cube_start_x_range[1] - self.cube_start_x_range[0]
        y_scope = self.cube_start_y_range[1] - self.cube_start_y_range[0]

        for i in range(self.cube_total_count):
            progress = (i + 40) / (self.cube_total_count + 40)

            start_x_range = int(center_x - progress * x_scope / 2), int(center_x + progress * x_scope / 2)
            start_y_range = int(center_y - progress * y_scope / 2), int(center_y + progress * y_scope / 2)

            x = np.random.randint(*start_x_range)
            y = np.random.randint(*start_y_range)

            cur_cube = Cube(x, y, cur_cats[i], self.stuffs)
            self.cubes.append(cur_cube)

        self.update_stacked_relationship()
        self.top_cubes = self.get_top_cubes()

    def update_stacked_relationship(self):
        """
        更新堆叠关系
        :return:
        """
        for cur_index in range(len(self.cubes)):
            cur_cube = self.cubes[cur_index]
            cur_cube.bottoms.clear()
            cur_cube.tops.clear()
            for pre_index in range(cur_index):
                pre_cube = self.cubes[pre_index]
                # 覆盖了之前的cube
                if self.cal_iou(pre_cube, cur_cube) > self.cover_iou_thresh:
                    cur_cube.bottoms.append(pre_cube)
                    pre_cube.tops.append(cur_cube)

    def get_top_cubes(self):
        """
        初始化
        :return:
        """
        top_cubes = []
        for cube in self.cubes:
            if not cube.tops:
                top_cubes.append(cube)
        return top_cubes

    def draw_cubes(self):
        """
        画当前的game
        :return:
        """
        bg = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # main area
        for cube in self.cubes:
            tops_count = len(cube.tops)
            if tops_count == 0:
                cube_vis_img = cube.stuff.bright_img
            elif tops_count == 1:
                cube_vis_img = cube.stuff.bright_dark_img
            else:
                cube_vis_img = cube.stuff.dark_img
            cube_vis_img_copy = cube_vis_img.copy()

            valid = cube_vis_img[..., 3] != 0  # 非透明
            cube_vis_img_copy[..., 3][valid] = 255

            bg[cube.y:cube.y + cube.h, cube.x:cube.x + cube.w][valid] = cube_vis_img_copy[valid]

        # slots area
        for i in range(self.max_slots_cubes):
            x, y = self.slots_positions[i]
            if i < len(self.slot_cubes):
                cube = self.slot_cubes[i]
                cube_vis_img = cube.stuff.bright_img
                cube_vis_img_copy = cube_vis_img.copy()

                valid = cube_vis_img[..., 3] != 0  # 非透明
                cube_vis_img_copy[..., 3][valid] = 255

                bg[y:y + CUBE_HEIGHT, x:x + CUBE_WIDTH][valid] = cube_vis_img_copy[valid]
            else:
                cv2.rectangle(bg, pt1=(x, y), pt2=(x + CUBE_WIDTH, y + CUBE_HEIGHT), color=(255, 0, 255, 255), )
        return bg

    def is_win(self):
        """
        判断当前是否赢了。
        :return:
        """
        return len(self.cubes) == 0 and len(self.slot_cubes) == 0

    def insert_to_slots(self, insert_cube: Cube):
        """
        把一个方块插入到槽中。如果有同样的块，则插入，如果没有同样类别的，则补充到后面；
        :param insert_cube:
        :return:
        """
        insert_index = -1
        for index, cube in enumerate(self.slot_cubes):
            if cube.category == insert_cube.category:
                insert_index = index
                break

        if insert_index != -1:
            self.slot_cubes.insert(insert_index, insert_cube)
        else:
            self.slot_cubes.append(insert_cube)

    def shuffle_cur_cubes_category(self):
        """
        随机打乱当前方块们的类别
        :return:
        """
        cats = [cube.category for cube in self.cubes]
        np.random.shuffle(cats)

        for idx, cube in enumerate(self.cubes):
            cube.category = cats[idx]
            cube.stuff = self.stuffs[cube.category]

    def move_up_cubes(self):
        """
        上移若干方块
        :return:
        """
        # 如果是1，2，3个都可以移动到
        move_up_nums = min(self.max_move_up_cube_nums, len(self.slot_cubes))
        for i in range(move_up_nums)[::-1]:  # 从后往前删除，避免就地删除造成的索引越界
            cube = self.slot_cubes[i]
            start_x, start_y = self.move_up_positions[i]
            cur_pos_cubes = self.move_up_cubes_dict[i]

            start_x += len(cur_pos_cubes) * self.move_up_cubes_tile_offset_x
            start_y += len(cur_pos_cubes) * self.move_up_cubes_tile_offset_y

            cube.x = start_x
            cube.y = start_y

            self.slot_cubes.remove(cube)
            self.cubes.append(cube)
            self.move_up_cubes_dict[i].append(cube)

        self.update_stacked_relationship()

    def remove_three_same_cubes(self):
        """
        如果有三个连续方块类别一致，则移除
        :return:
        """
        delete_start = -1
        for index in range(len(self.slot_cubes[:-2])):
            first_cat = self.slot_cubes[index].category
            second_cat = self.slot_cubes[index + 1].category
            third_cat = self.slot_cubes[index + 2].category
            if first_cat == second_cat == third_cat:
                delete_start = index

        if delete_start != -1:
            self.slot_cubes.remove(self.slot_cubes[delete_start + 2])
            self.slot_cubes.remove(self.slot_cubes[delete_start + 1])
            self.slot_cubes.remove(self.slot_cubes[delete_start + 0])

    def handle_click_game(self, click_x: int, click_y: int) -> bool:
        """
        遍历当前顶层方块。处理点击界面，如果点击到了顶层块，则返回true，否则返回false
        :param click_x:
        :param click_y:
        :return:
        """
        for cube in self.top_cubes:
            if cube.x < click_x < cube.x + cube.w and cube.y < click_y < cube.y + cube.h:
                for bottom_cube in cube.bottoms:
                    bottom_cube.tops.remove(cube)
                    if not bottom_cube.tops:  # 某个方块的上层方块全部被移除了
                        self.top_cubes.append(bottom_cube)

                self.remove_cube(cube)
                self.insert_to_slots(cube)
                self.remove_three_same_cubes()

                return True
        return False

    def remove_cube(self, cube: Cube):
        """
        从主区域和上移区域中移除某个方块。如果存在的话
        :param remove_cube:
        :return:
        """
        self.cubes.remove(cube)
        self.remove_from_move_up_area(cube)
        self.top_cubes.remove(cube)

    def remove_from_move_up_area(self, remove_cube: Cube):
        """
        从上移位置删除方块
        :param remove_cube:
        :return:
        """
        for index in self.move_up_cubes_dict:
            cubes = self.move_up_cubes_dict[index]
            for cube in cubes:
                if cube == remove_cube:
                    cubes.remove(cube)
                    return

    def is_game_over(self):
        """
        判断是否游戏结束
        :return:
        """
        return len(self.slot_cubes) == self.max_slots_cubes

    @staticmethod
    def cal_iou(cube1: Cube, cube2: Cube):
        """
        计算两个方块的IoU,intersection over union
        :param cube1:
        :param cube2:
        :return:
        """
        min_x = min(cube1.x, cube2.x)
        max_x = max(cube1.x + cube1.w, cube2.x + cube2.w)

        x_overlap = (cube1.w + cube2.w) - (max_x - min_x)

        min_y = min(cube1.y, cube2.y)
        max_y = max(cube1.y + cube1.h, cube2.y + cube2.h)

        y_overlap = (cube1.h + cube2.h) - (max_y - min_y)

        if x_overlap <= 0 or y_overlap <= 0:
            return 0

        return x_overlap * y_overlap / (cube1.w * cube1.h + cube2.w * cube2.h)


class MainWindow(QWidget):
    def __init__(self):
        """
        主窗体初始化
        """
        super(MainWindow, self).__init__()

        self.pic_box = PicBox()
        self.game_region = QFrame()
        self.button_region = QFrame()

        self.button_top_margin = 50
        self.button_vertical_interval = 20
        self.button_left_right_margin = 20

        self.button_width = 120
        self.button_height = 45

        self.pic_box_width = 450
        self.pic_box_height = 720

        self.resize(self.pic_box_width + 2 * self.button_left_right_margin + self.button_width, self.pic_box_height)

        self.remainder_label = QLabel()
        self.move_up_cubes_button = QPushButton()
        self.undo_button = QPushButton()
        self.shuffle_button = QPushButton()
        self.restart_button = QPushButton()

        self.init_game_region()
        self.init_pic_box()  # pic box属于game region
        self.init_button_region()

        self.game = None
        self.restart_game()

        print('init game')

    def init_button_region(self):
        """
        初始化按钮区域
        :return:
        """
        vertical_index = 0
        self.button_region.setParent(self)
        self.button_region.move(self.pic_box.width(), 0)
        self.button_region.resize(self.button_width + 2 * self.button_left_right_margin, self.height())

        self.remainder_label.setParent(self.button_region)
        self.remainder_label.resize(self.button_width, self.button_height)
        self.remainder_label.move(self.button_left_right_margin,
                                  self.button_top_margin + (
                                          self.button_vertical_interval + self.button_height) * vertical_index)
        self.remainder_label.setText('剩余数量：')

        vertical_index += 1
        self.move_up_cubes_button.setParent(self.button_region)
        self.move_up_cubes_button.resize(self.button_width, self.button_height)
        self.move_up_cubes_button.move(self.button_left_right_margin,
                                       self.button_top_margin + (
                                               self.button_vertical_interval + self.button_height) * vertical_index)
        self.move_up_cubes_button.setText('移除3个方块')
        self.move_up_cubes_button.clicked.connect(self.move_up_cubes)

        # vertical_index += 1
        # self.undo_button.setParent(self.button_region)
        # self.undo_button.resize(self.button_width, self.button_height)
        # self.undo_button.move(self.button_left_right_margin,
        #                       self.button_top_margin + (
        #                               self.button_vertical_interval + self.button_height) * vertical_index)
        # self.undo_button.setText('撤销')  # 暂时不实现

        vertical_index += 1
        self.shuffle_button.setParent(self.button_region)
        self.shuffle_button.resize(self.button_width, self.button_height)
        self.shuffle_button.move(self.button_left_right_margin,
                                 self.button_top_margin + (
                                         self.button_vertical_interval + self.button_height) * vertical_index)
        self.shuffle_button.setText('打乱')
        self.shuffle_button.clicked.connect(self.shuffle_cur_cubes_category)

        vertical_index += 1
        self.restart_button.setParent(self.button_region)
        self.restart_button.resize(self.button_width, self.button_height)
        self.restart_button.move(self.button_left_right_margin,
                                 self.button_top_margin + (
                                         self.button_vertical_interval + self.button_height) * vertical_index)
        self.restart_button.clicked.connect(self.restart_game)
        self.restart_button.setText('重启')

    def shuffle_cur_cubes_category(self):
        """
        随机打乱方块的类别
        :return:
        """
        self.game.shuffle_cur_cubes_category()
        self.show_cur_game()

    def restart_game(self):
        """
        重启游戏
        :return:
        """
        self.game = Game()
        self.game.random_cubes()
        self.show_cur_game()

    def show_cur_game(self):
        """
        可视化当前游戏。包括主界面和按钮显示当前数量
        :return:
        """
        img = self.game.draw_cubes()
        self.show_image_on_pic_box(img)
        self.show_cube_num()

    def move_up_cubes(self):
        """
        将若干方块移动到上方。并且遍历所有方块更新顶层方块。
        :return:
        """
        self.game.move_up_cubes()
        self.game.top_cubes = self.game.get_top_cubes()
        self.show_cur_game()

    def show_image_on_pic_box(self, img: np.ndarray):
        """
        将图片显示在图相框中
        Args:
            img:

        Returns:

        """
        h, w = img.shape[:2]
        if h != self.pic_box_height and w != self.pic_box_width:
            img = cv2.resize(img, dsize=(self.pic_box_width, self.pic_box_height), interpolation=cv2.INTER_LINEAR)

        q_image = QImage(img.tobytes(), self.pic_box_width, self.pic_box_height, QImage.Format_RGBA8888)
        q_pixel_map = QPixmap.fromImage(q_image)
        self.pic_box.setPixmap(q_pixel_map)

    def init_game_region(self):
        """
        初始化game区域
        :return:
        """
        self.game_region.setParent(self)
        self.game_region.move(0, 0)
        self.game_region.resize(self.pic_box_width, self.pic_box_height)

    def init_pic_box(self):
        """
        初始化图片框
        :return:
        """
        self.pic_box.setParent(self.game_region)
        self.pic_box.setStyleSheet('border:10px solid #242424;')
        self.pic_box.move(0, 0)
        self.pic_box.resize(self.pic_box_width, self.pic_box_height)

        self.pic_box.click_finished.connect(self.click_pic_box)

    def click_pic_box(self, event: QtGui.QMouseEvent):
        """
        处理点击图片框的事件
        :param event:
        :return:
        """
        click_x = event.x()
        click_y = event.y()

        game_x, game_y = self.map2game(click_x, click_y)

        # 如果没有点击到顶层块
        if not self.game.handle_click_game(game_x, game_y):
            return

        self.show_cur_game()

        if self.game.is_game_over():
            msg_box = QMessageBox(self)
            msg_box.information(self, '游戏结束!', '你输了', buttons=QMessageBox.Yes)
            self.restart_game()
            return
        if self.game.is_win():
            msg_box = QMessageBox(self)
            msg_box.information(self, 'Win!', '你赢了', buttons=QMessageBox.Yes)
            self.restart_game()

    def show_cube_num(self):
        """
        显示当前剩余方块数量
        :return:
        """
        cube_num = len(self.game.cubes)
        self.remainder_label.setText(f'剩余：{cube_num} 个')

    def map2game(self, x: int, y: int):
        """
        将图片框的点击坐标map到game内
        :param x:
        :param y:
        :return:
        """
        game_x = x / self.pic_box_width * self.game.width
        game_y = y / self.pic_box_height * self.game.height
        return game_x, game_y


if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    yang_game = MainWindow()
    yang_game.show()
    sys.exit(myapp.exec_())
