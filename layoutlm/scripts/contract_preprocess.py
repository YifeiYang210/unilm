import lib_pdf
import json
import shutil
import os
import glob
from transformers import BertTokenizer

pdf_dir = r"./data/pdf"
pdf_folder = os.listdir(pdf_dir)


def word_mul_factor_equal_entity(nw, ne):
    thresh = 5
    return ne / 1.5 - thresh < nw < ne / 1.5 + thresh


def text_belong_to_the_label_entity_by_box(word, entity):
    # --根据坐标判断words是否属于实体
    word_bbox = word["bbox"]
    entity_bbox = entity["points"][0] + entity["points"][1]
    # --如果bbox的x1和y2符合1.5倍定律，那么则返回true
    if word_mul_factor_equal_entity(word_bbox[0], entity_bbox[0]) and \
            word_mul_factor_equal_entity(word_bbox[3], entity_bbox[3]):
        return True
    else:
        return False


def write_dataset():
    base_dir = r"./data/contract2020"
    base_folder = os.listdir(base_dir)
    with open("out.txt", "w", encoding="utf-8") as fw:
        # --遍历文件夹中的每个pdf文件
        for file in pdf_folder:
            pdf_src = os.path.join(pdf_dir, file)
            pdf_name = file.split(".")[0]

            # --开始遍历每个pdf文件多个页面对应的json文件，将json文件重新索引到页面对应的顺序位置
            related_file_list = glob.glob(pdf_src[:-4].replace("pdf", "contract2020") + "*")
            page_num = (len(related_file_list) - 1) // 2

            json_src_list = []
            for file_path in related_file_list:
                if file_path[-4:] == "json":
                    start_id = len(pdf_name)

                    # --如果有超过10张图片，条件是[10-99]
                    if page_num >= 10:
                        # --如果id是[0-9]，变成[00-09]方便后续排序
                        if not (file[start_id + 1] + file[start_id + 2]).isdigit():
                            json_path = os.path.join(base_dir, pdf_name[:-1] + pdf_name[-1].zfill(2) + ".json")
                            json_src_list.append(json_path)
                    else:
                        json_src_list.append(file_path)
            json_src_list.sort()

            pdf_raw = lib_pdf.read_pdf(pdf_src)

            for page_id in range(0, page_num):
                json_src_path = json_src_list[page_id]
                with open(json_src_path, "r", encoding="utf-8") as fr:
                    label_data = json.load(fr)

                    pdf_one_page = pdf_raw.pages[page_id]

                    # --读取单字信息
                    single_sorted_char_list = lib_pdf.get_single_chars_location_and_text(pdf_one_page)

                    # --得到行信息
                    words_info = lib_pdf.get_words_locations(single_sorted_char_list)

                    # --转化为word为行的数据集
                    for word in words_info:
                        label_name = ""
                        for entity in label_data["shapes"]:
                            if entity["label"] == "Supplier" or entity["label"] == "Table":
                                continue
                            if text_belong_to_the_label_entity_by_box(word, entity):
                                label_name = entity["label"]
                                break
                            else:
                                label_name = "O"
                        fw.write(
                            word["text"] + "\t\t\t" + label_name + "\t\t\t" + ",".join(
                                [str(x) for x in word["bbox"]]) + "\n")
                    fw.write("\n")


def debug_rename_file_path_format():
    base_dir = r"./data/contract2020"
    base_folder = os.listdir(base_dir)
    for file in base_folder:
        new_file_name = file.replace("-", "_")
        new_file_src = os.path.join(base_dir, new_file_name)
        old_file_src = os.path.join(base_dir, file)
        os.rename(old_file_src, new_file_src)


def debug_move_pdf_to_search_index():
    base_dir = r"./data/contract2020"
    base_folder = os.listdir(base_dir)
    for file in base_folder:
        if file[-3:].lower() == "pdf":
            print(file)
            shutil.copyfile(os.path.join(base_dir, file), os.path.join(pdf_dir, file))
    print("ok")


def write_label():
    s = set()
    with open("./data/test.txt", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = line.split("\t\t")
            if len(raw) != 3:
                assert raw, "fuck"
            label = raw[1]
            s.add(label)
    with open("labels.txt", "w", encoding="utf-8") as fw:
        for each in s:
            fw.write(each + "\n")

    print("ok")


def seg_file(file_path, tokenizer, max_len, mode):
    # --分词长度计数
    subword_len_counter = 0
    output_path = f"./{mode}.txt"
    with open(file_path, "r", encoding="utf-8") as f_p, open(
            output_path, "w", encoding="utf-8") as fw_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue
            token = line.split("\t\t")[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                subword_len_counter = 0
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")


tokenizer = BertTokenizer.from_pretrained(
    "./mymodel", do_lower_case=True
)
mode = "test"
file_path = f"./data/{mode}.txt"
max_len = 510


# seg_file(file_path, tokenizer, max_len, mode)3

def debug_what_is_bad_case_after_training():
    with open(r"./output/test_predictions.txt", "r", encoding="utf-8") as fp, open(r"./data/test.txt", "r",
                                                                               encoding="utf-8") as ft:
        for pline, tline in zip(fp, ft):
            if not pline.strip():
                continue
            p_raw = pline.split(" ")
            t_raw = tline.split("\t\t")
            if p_raw[1].strip() != t_raw[1]:
                print(p_raw, t_raw, sep=" | ")


debug_what_is_bad_case_after_training()