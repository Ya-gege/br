#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@Author :DEV_YH
@File :hashi.py
@Time :2024/3/3 13:07
@Description: 
"""
import sys

import numpy as np

sys.setrecursionlimit(3000)

# East South West North
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DOT = '.'
TOTAL_BRIDGE = 0
MAXIMUM_PAIR_BRIDGES = 3
# 1- West or East
# 2- North or South
BRIDGE_MAP = {-1: '-',
              -2: '=',
              -3: 'E',
              -4: '|',
              -5: '"',
              -6: '#'}


class Neighbor:
    def __init__(self, n, e, s, w):
        self.north = n
        self.east = e
        self.south = s
        self.west = w

    def to_list(self):
        l = list()
        if self.north is not None:
            l.append(self.north)
        if self.south is not None:
            l.append(self.south)
        if self.west is not None:
            l.append(self.west)
        if self.east is not None:
            l.append(self.east)
        return l


class NeighborPair:
    def __init__(self, neighbor: Neighbor):
        self.neighbor = neighbor
        self.nbridge = 0


class Node:
    def __init__(self, row, col, neighbor_pair=None, val=0):
        self.row = row
        self.col = col
        self.neighbor_pair = neighbor_pair
        self.val = val
        self.nline = 0

    def can_be_built(self, n):
        return self.nline + n <= self.val

    def finish_built(self):
        return self.nline == self.val

    def add_bridge(self, n):
        self.nline += n

    def decrease_bridge(self, n):
        self.nline -= n

    def over_size(self):
        return self.nline > self.val


def scan_map():
    text = []
    for line in sys.stdin:
        if line == 'ok' + '\n':
            break
        row = []
        for ch in line:
            n = ord(ch)
            if 48 <= n <= 57:  # between '0' and '9'
                row.append(n - 48)
            elif 97 <= n <= 122:  # between 'a' and 'z'
                row.append(n - 87)
            elif ch == '.':
                row.append(0)
        text.append(row)

    nrow = len(text)
    ncol = len(text[0])

    mat = np.zeros((nrow, ncol), dtype=np.int32)
    for r in range(nrow):
        for c in range(ncol):
            mat[r, c] = text[r][c]

    sys.stdin.close()
    return nrow, ncol, mat


# Find the start node
def find_start(mat, nrow, ncol):
    for i in range(nrow):
        for j in range(ncol):
            if DOT == mat[i, j]:
                return i, j
    return -1, -1


def get_or_add_node(k, node_map, i, j, val):
    if k not in node_map:
        node_map[k] = Node(i, j, None, val)
    return node_map[k]


def find_neighbor(mat, row, col, node_map):
    nb = Neighbor(None, None, None, None)
    for direct in DIRECTIONS:
        r = row + direct[0]
        c = col + direct[1]
        while check_border(r, c, mat):
            node_k = str(r) + str(c)
            if mat[r, c] != 0:
                node = get_or_add_node(node_k, node_map, r, c, mat[r, c])
                k = str(direct[0]) + str(direct[1])
                if k == '10':
                    nb.south = node
                if k == '-10':
                    nb.north = node
                if k == '0-1':
                    nb.west = node
                if k == '01':
                    nb.east = node
                break
            r = r + direct[0]
            c = c + direct[1]
    return nb


# Check the border of mat
def check_border(nrow, ncol, mat):
    return 0 <= nrow < len(mat) and 0 <= ncol < len(mat[0])


def init_nodes(mat, nrow, ncol):
    node_map = dict()
    for i in range(nrow):
        for j in range(ncol):
            k = str(i) + str(j)
            if mat[i, j] != 0:
                node = get_or_add_node(k, node_map, i, j, mat[i, j])
                node.neighbor_pair = NeighborPair(find_neighbor(mat, i, j, node_map))
    return node_map


def build_bridge(pre_node, node, n, bridge_map):
    dx = node.row - pre_node.row
    dy = node.col - pre_node.col
    # Build bridge
    if dx == 0:
        start = min(pre_node.col, node.col)
        end = max(pre_node.col, node.col)
        bridge_map[node.row, start + 1:end] += n
    if dy == 0:
        start = min(pre_node.row, node.row)
        end = max(pre_node.row, node.row)
        bridge_map[start + 1:end, node.col] += n


def sum_bridge(bridge_map):
    return np.sum(bridge_map)


def finish_built(node_map):
    for node in node_map.values():
        if node.val != node.nline:
            return False
    return True


def limit_check(pre_node, node, bridge_num, bridge_map):
    dx = node.row - pre_node.row
    dy = node.col - pre_node.col
    curr = 0
    # Build bridge
    if dx == 0:
        start = min(pre_node.col, node.col)
        curr = bridge_map[node.row, start + 1]
    if dy == 0:
        start = min(pre_node.row, node.row)
        curr = bridge_map[start + 1, node.col]
    return curr + bridge_num <= 3


def cross_product(x1, y1, x2, y2):
    return x1 * y2 - y1 * x2


def cal_vector(x1, y1, x2, y2):
    return x2 - x1, y2 - y1


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x) + '-' + str(self.y)


class Line:
    def __init__(self, start_point: Point, end_point: Point, f=None, to=None, val=0):
        self.start_point = start_point
        self.end_point = end_point
        self.f = f
        self.to = to
        self.val = 0

    def get_key(self):
        return str(self.start_point) + '=' + str(self.end_point)


def cross_check_base(pre_node, node, built_map: dict, f: Point, t: Point):
    x1 = pre_node.row
    y1 = pre_node.col
    x2 = node.row
    y2 = node.col
    # AB
    xx1, yy1 = cal_vector(x1, y1, x2, y2)
    for line in built_map.values():
        start = line.start_point
        end = line.end_point
        # # CD
        # xx2, yy2 = cal_vector(start.x, start.y, end.x, end.y)
        # AC
        xx3, yy3 = cal_vector(x1, y1, start.x, start.y)
        # # BD
        # xx4, yy4 = cal_vector(x2, y2, end.x, end.y)
        # AD
        xx5, yy5 = cal_vector(x1, y1, end.x, end.y)

        # AB cross AC
        cp1 = cross_product(xx1, yy1, xx3, yy3)
        # AB cross AD
        cp2 = cross_product(xx1, yy1, xx5, yy5)

        # Cross
        if cp1 * cp2 < 0:
            return False
        # Collinear or parallel
        if cp1 * cp2 == 0:
            line_from = line.f
            line_to = line.to
            if start.x == pre_node.row and start.y == pre_node.col:
                if not (f.x == line_from.x and f.y == line_from.y and t.x == line_to.x and t.y == line_to.y):
                    return False
    return True


def get_bridge_point(pre_node: Node, node: Node):
    row = pre_node.row
    col = pre_node.col
    # Start
    neighbor = pre_node.neighbor_pair.neighbor
    if neighbor.north == node:
        return Node(row - 1, col)
    if neighbor.south == node:
        return Node(row + 1, col)
    if neighbor.west == node:
        return Node(row, col - 1)
    if neighbor.east == node:
        return Node(row, col + 1)


def cross_check(pre_node, node, built_map: dict):
    if len(built_map) == 0:
        return True
    f_point = Point(pre_node.row, pre_node.col)
    t_point = Point(node.row, node.col)
    start_node = get_bridge_point(pre_node, node)
    end_node = get_bridge_point(node, pre_node)
    return cross_check_base(start_node, end_node, built_map, f_point, t_point)


def add_built_map(pre_node, node, built_map: dict, n):
    start_node = get_bridge_point(pre_node, node)
    end_node = get_bridge_point(node, pre_node)
    start = Point(start_node.row, start_node.col)
    end = Point(end_node.row, end_node.col)
    # 有可能重复建桥
    line = Line(start, end, Point(pre_node.row, pre_node.col), Point(node.row, node.col), 0)
    k = line.get_key()
    if k not in built_map.keys():
        built_map[k] = line
    line_cull = built_map[k]
    line_cull.val += n


def remove_built_map(pre_node, node, built_map: dict, n):
    start_node = get_bridge_point(pre_node, node)
    end_node = get_bridge_point(node, pre_node)
    start = Point(start_node.row, start_node.col)
    end = Point(end_node.row, end_node.col)
    line = Line(start, end)
    k = line.get_key()
    line = built_map[k]
    line.val -= n
    if line.val == 0:
        built_map.pop(k)


def print_path(node, next_node, num):
    x1 = node.row
    y1 = node.col
    x2 = next_node.row
    y2 = next_node.col
    s = str(x1) + '-' + str(y1) + '=' + str(x2) + '-' + str(y2) + '==' + str(num)
    print(s)


def try_build(pre_node: Node, node: Node, bridge_map, node_map, built_map, idx):
    # if pre_node is not None:
    #     # Bridges are not allowed to cross each other
    #     if not cross_check(pre_node, node, bridge_map):
    #         return False
    #     build_bridge(pre_node, node, nbridge, bridge_map)

    # Bridges n <= number of node
    if node.over_size():
        return False

    # All bridges build ok
    if finish_built(node_map):
        return True

    # Can build bridges
    neighbor_list = node.neighbor_pair.neighbor.to_list()
    for next_node in neighbor_list:
        if pre_node is not None and next_node == pre_node:
            continue
        for bridge_num in range(1, 4):
            # Each node can be built
            if node.can_be_built(bridge_num) \
                    and next_node.can_be_built(bridge_num) \
                    and cross_check(node, next_node, built_map) \
                    and limit_check(node, next_node, bridge_num, bridge_map):
                # Try build bridges
                # print_path(node, next_node, bridge_num)
                build_bridge(node, next_node, bridge_num, bridge_map)
                # Add bridge
                add_built_map(node, next_node, built_map, bridge_num)
                node.add_bridge(bridge_num)
                next_node.add_bridge(bridge_num)
                if try_build(node, next_node, bridge_map, node_map, built_map, idx + 1):
                    return True
                # Backtrack
                node.decrease_bridge(bridge_num)
                next_node.decrease_bridge(bridge_num)
                # Remove bridge
                build_bridge(node, next_node, -1 * bridge_num, bridge_map)
                remove_built_map(node, next_node, built_map, bridge_num)
    return False


def main():
    nrow, ncol, mat = scan_map()
    node_map = init_nodes(mat, nrow, ncol)
    bridge_map = np.zeros((nrow, ncol), dtype=np.int32)
    built_map = dict()
    for i in range(nrow):
        for j in range(ncol):
            k = str(i) + str(j)
            if k not in node_map.keys():
                continue
            node = node_map[k]
            try_build(None, node, bridge_map, node_map, built_map, 0)
            break
    print(1)


if __name__ == '__main__':
    main()
