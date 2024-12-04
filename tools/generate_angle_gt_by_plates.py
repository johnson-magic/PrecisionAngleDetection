import os
import sys
import json
import tqdm
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.angle_utils import calculate_rotation_angle_box


def arg_parser():
    parser = argparse.ArgumentParser(description="生成角度真值")
    
    # 添加参数
    parser.add_argument('--test_dataset_labels_root', type=str, required=True, help='测试集目标检测真值路径')

    return parser


def read_and_filter_label_file(label_file_path: str) -> list:
    """读取原始目标检测标定文件，并filter、返回需要的label信息
    
    Args:
        label_file_path: str, 原始目标标定文件，其为多行c x1 y1 x2 y2 x3 y3 x4 y4协议的txt文件
    
    Return:
        list, plates的检测label信息[1, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    labels = []  # 用于存储结果的列表
    with open(label_file_path, 'r') as file:
        for line in file:
            # 使用strip()去除行末的换行符，然后以空格分割
            label = line.strip().split()
            # 将分割出的字符串转换为浮点数并添加到列表中
            labels.append([float(l) for l in label])  # label表示[c, x1, y1, x2, y2, x3, y3, x4, y4]
    
    filter_labels = [label for label in labels if label[0] == 1]  # “1”对应类别为plates, 我们仅需要plates
    
    assert len(filter_labels) == 1  # 必须严格要求一张图片中，仅有一个plates, 通过断言及时发现异常
    
    return filter_labels[0]


def generate_angle_gt_by_plates(dataset_labels_root: str) -> None:
    """通过plates的坐标label信息生成数据集的angle标签。
    angle标签存放路径为dataset_labels_root/angle_gts.json。
    
    Args:
        dataset_labels_root: str, 数据集标签文件root目录。
    """
    angle_gts_dict = {}
    
    for txt_file in tqdm.tqdm(os.listdir(dataset_labels_root)):
        if not txt_file.endswith(".txt"):
            continue
        
        txt_full_path = os.path.join(dataset_labels_root, txt_file)
        plates_obj_label = read_and_filter_label_file(txt_full_path)
        plates_obj_label_coors = plates_obj_label[1:]  # 去除第一位的类别
        
        plates_obj_label_coors = [(plates_obj_label_coors[i], plates_obj_label_coors[i + 1]) for i in range(0, len(plates_obj_label_coors), 2)]
        angle_gt = calculate_rotation_angle_box(plates_obj_label_coors)
        angle_gts_dict[txt_file] = angle_gt
    
    gt_save_full_path = os.path.join(dataset_labels_root, "angle_gts.json")
    with open(gt_save_full_path, "w") as fw:
        json.dump(angle_gts_dict, fw, ensure_ascii = False, indent = 4)
    

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    generate_angle_gt_by_plates(args.test_dataset_labels_root)
    
    # e.g. python 
    # python3 tools/generate_angle_gt_by_plates.py --test_dataset_labels_root ./../datasets/obb-project/labels/val