# -*- coding: utf-8 -*-
import json, torch, cv2, time, os, tempfile
import numpy as np
import pdfplumber, pdfminer, wand.image
import logging
from PIL import Image
from io import BytesIO
from pdfminer.layout import *

logging.getLogger('pdfminer').setLevel(logging.ERROR)


def read_pdf(temp_file):
    pdf_raw = pdfplumber.open(temp_file)
    return pdf_raw


def get_one_page_image(temp_file, one_pdf_page, resolution=300):
    page_path = "{0}[{1}]".format(temp_file, one_pdf_page.page_number - 1)
    with wand.image.Image(filename=page_path, resolution=resolution) as img:
        if img.alpha_channel:
            img.background_color = wand.image.Color('white')
            img.alpha_channel = 'background'
        if True:
            im = Image.open(BytesIO(img.make_blob(format='jpg')))
            if "transparency" in im.info:
                converted = im.convert("RGBA").convert("RGB")
            else:
                converted = im.convert("RGB")
            return converted


def get_words_locations(chars_info, crop_bbox=None):
    # --目标：把PDF单字聚类成字符串
    # --问题：（1）不能按绝对Y坐标，因为会有偏差
    # --总体思路，
    # 1）首先，对所有单字进行组合，把可能一行的放在一个数组lines_info中，从第一个字符开始，找到这个字符可能在同行的，Y投影方向重合80%，然后进行X排序
    # 2）然后，对lines_info进行处理，判断两个字符大小是否差不多，并且间距正常，组成words_info

    # --得到行信息
    lines_info = get_lines_from_single_chars(chars_info, crop_bbox=crop_bbox)

    # --得到words_info
    words_info = get_words_from_lines(lines_info)

    return words_info


def debug_words_info(words_info_from_pdf_text):
    line_row_id = 0
    for word in words_info_from_pdf_text:
        if word['row_id'] != line_row_id:
            print()
            print(word['row_id'], '>>', end=' ')
            line_row_id = word['row_id']
        print(word['text'], word['bbox'], end=' | ')
    print()


# --根据行信息得到分开的words数组信息，每个words包括text、bbox、row_id信息，并已经根据y，x坐标排序
def get_words_from_lines(lines_info, F_magic_split_str='$__--__$', F_min_interval_h_ratio=0.36):
    words_info = []

    for line_row_id, line in enumerate(lines_info):
        # --先通过字符间距离把字符分开得到words
        str_line_with_split_char = get_str_line_witch_split_char(line, F_magic_split_str, F_min_interval_h_ratio)
        words_list = str_line_with_split_char.split(F_magic_split_str)

        # --得到words的位置，这个后续定位行列化很重要
        start_pos = 0
        for word in words_list:
            new_row = {}
            # --保持兼容，置信率和单字位置
            new_row['scores'] = [1.0] * len(word)
            new_row['single_position'] = None
            new_row['text'] = word
            new_row['row_id'] = line_row_id + 1
            x1 = min(line[start_pos:start_pos + len(word)], key=lambda row: row['bbox'][0])['bbox'][0]
            y1 = min(line[start_pos:start_pos + len(word)], key=lambda row: row['bbox'][1])['bbox'][1]
            x2 = max(line[start_pos:start_pos + len(word)], key=lambda row: row['bbox'][2])['bbox'][2]
            y2 = max(line[start_pos:start_pos + len(word)], key=lambda row: row['bbox'][3])['bbox'][3]
            new_row['bbox'] = [x1, y1, x2, y2]
            new_row['4_box'] = [x1, y1, x2, y1, x2, y2, x1, y2]
            words_info.append(new_row)
            start_pos += len(word)

    sorted_words_info = sorted(words_info, key=lambda row: row['row_id'])
    return sorted_words_info


# --针对一行，分解得到str
def get_str_line_witch_split_char(line, magic_split_str, min_interval_h_ratio):
    str_line_with_split_char = line[0]['text']
    for idx in range(1, len(line)):
        last_char = line[idx - 1]
        char = line[idx]

        # --最小间隔，等于字符宽度*min_interval_h_ratio
        min_h = int(min_interval_h_ratio * ((max(int(char['width']), int(last_char['width'])))))

        # --如果字符间隔大于最小间隔，就认为是分开了，加上一个特定字符串，便于后续分开
        if (char['bbox'][0] - last_char['bbox'][2]) > min_h:
            str_line_with_split_char = str_line_with_split_char + magic_split_str + char['text']
        else:
            str_line_with_split_char = str_line_with_split_char + char['text']
    return str_line_with_split_char


# --根据单字信息，得到行信息，多行组成一个数组，每行包括即单字数组，单字按照x坐标排序
def get_lines_from_single_chars(raw_chars_info, crop_bbox, F_total_h_ratio=1.4):
    chars_dict = {}
    lines_info = []

    # --只把特定的区域计算进来
    if crop_bbox is not None:
        chars_info = []
        for char in raw_chars_info:
            # --判断char中心点在crop区域
            center_x = char['bbox'][0] + 0.5 * char['width']
            center_y = char['bbox'][1] + 0.5 * char['height']
            if center_x >= crop_bbox[0] and center_y >= crop_bbox[1] and center_x <= crop_bbox[2] and center_y <= \
                    crop_bbox[3]:
                chars_info.append(char)
    else:
        chars_info = raw_chars_info

    for idx in range(0, len(chars_info)):
        if str(idx) not in chars_dict:  # char_id还没被用
            char = chars_info[idx]
            chars_dict[str(idx)] = len(lines_info)  # 映射ID，即char_id -> line_id
            new_line = [char]  # 另起一行
            for j in range(idx + 1, len(chars_info)):
                next_char = chars_info[j]
                if judge_two_char_same_line(char, next_char, F_total_h_ratio):
                    chars_dict[str(j)] = len(lines_info)
                    new_line.append(next_char)

            # --新的一行按照x排序
            sorted_new_line = sorted(new_line, key=lambda row: row['bbox'][0])
            lines_info.append(sorted_new_line)

    return lines_info


# --判断是否在同一直线，根据Y方向重合度判断即可
def judge_two_char_same_line(char, next_char, total_h_ratio):
    # --2个字符串bbox的高度之和小于一个阈值，1.4倍最大H
    total_h = max(next_char['bbox'][3], char['bbox'][3]) - min(next_char['bbox'][1], char['bbox'][1])
    return total_h <= (total_h_ratio * max(char['height'], next_char['height']))


def get_single_chars_location_and_text(page):
    text_infos = []
    for i in page.layout:
        if isinstance(i, LTChar):
            text_infos.append(i)

    text_list = []
    for text in text_infos:
        if len(text._text.strip()) > 0:
            new_row = {}
            new_row['bbox'] = [
                min(int(text.x0), int(text.x1)),
                min(int(page.height) - int(text.y0), int(page.height) - int(text.y1)),
                max(int(text.x0), int(text.x1)),
                max(int(page.height) - int(text.y0), int(page.height) - int(text.y1)),
            ]
            new_row['text'] = text._text
            new_row['width'] = abs(new_row['bbox'][2] - new_row['bbox'][0])
            new_row['height'] = abs(new_row['bbox'][3] - new_row['bbox'][1])
            text_list.append(new_row)

    sorted_char_list = sorted(text_list, key=lambda row: (row['bbox'][1], row['bbox'][0]))

    return sorted_char_list


def judge_table_img_one_page(one_pdf_page, F_min_w_ratio=0.65, F_min_h_ratio=0.15):
    # --判断该页是否可能是一个图像表格
    try:
        for r in one_pdf_page.objects['image']:
            if float(r['width']) > float(one_pdf_page.width) * float(F_min_w_ratio) or float(r['height']) > float(
                    one_pdf_page.height) * float(F_min_h_ratio):
                return True
    except:
        return False

    return False


def judge_have_line(page, line_width=2):
    lr = []
    for i in page.layout:
        # 去掉点
        if i.height < line_width and i.width < line_width:
            continue
        # 筛选线的类型
        if isinstance(i, LTRect) or isinstance(i, LTLine):
            # 去掉矩形
            if i.height > line_width and i.width > line_width:
                continue
            else:
                lr.append(i)

    return len(lr) > 8
