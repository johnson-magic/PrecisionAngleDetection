import os
import sys
import json
import tqdm
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.angle_utils import calculate_rotation_angle_box, calculate_center, calculate_rotation_angle_line


def arg_parser():
    parser = argparse.ArgumentParser(description="生成角度真值")
    
    # 添加参数
    parser.add_argument('--test_dataset_labels_root', type=str, required=True, help='测试集目标检测真值路径')
    parser.add_argument('--mode', type=str, required=True, default="all", help='one of plates, slide, big_circle, all')

    return parser


def read_and_filter_label_file(label_file_path: str, filter_id: int) -> list:
    """读取原始目标检测标定文件，并filter、返回需要的label信息
    
    Args:
        label_file_path: str, 原始目标标定文件，其为多行c x1 y1 x2 y2 x3 y3 x4 y4协议的txt文件
        filter_id: int, 需要的类别，0对应big_circle, 1对应plates, 2对应slide    
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
    
    filter_labels = [label for label in labels if label[0] == filter_id]  # 仅获取需要的目标
    
    assert len(filter_labels) == 1  # 必须严格要求一张图片中，仅有一个对应filter_id的目标, 通过断言及时发现异常
    
    return filter_labels[0]


def generate_angle_gt_by_plates(dataset_labels_root: str) -> None:
    """通过plates的坐标label信息生成数据集的angle标签。
    angle标签存放路径为dataset_labels_root/angle_gts_by_plates.json。
    
    Args:
        dataset_labels_root: str, 数据集标签文件root目录。
    """
    angle_gts_dict = {}
    
    for txt_file in tqdm.tqdm(os.listdir(dataset_labels_root)):
        if not txt_file.endswith(".txt"):
            continue
        
        txt_full_path = os.path.join(dataset_labels_root, txt_file)
        plates_obj_label = read_and_filter_label_file(txt_full_path, filter_id=1)  # 1表示plates的类别
        plates_obj_label_coors = plates_obj_label[1:]  # 去除第一位的类别
        
        plates_obj_label_coors = [(1200*plates_obj_label_coors[i], 1920*plates_obj_label_coors[i + 1]) for i in range(0, len(plates_obj_label_coors), 2)]
        angle_gt = calculate_rotation_angle_box(plates_obj_label_coors)
        angle_gts_dict[txt_file] = angle_gt
    
    gt_save_full_path = os.path.join(dataset_labels_root, "angle_gts_by_plates.json")
    with open(gt_save_full_path, "w") as fw:
        json.dump(angle_gts_dict, fw, ensure_ascii = False, indent = 4)

def generate_angle_gt_by_slide(dataset_labels_root: str) -> None:
    """通过slide的坐标label信息生成数据集的angle标签。
    angle标签存放路径为dataset_labels_root/angle_gts_by_slide.json。
    
    Args:
        dataset_labels_root: str, 数据集标签文件root目录。
    """
    angle_gts_dict = {}
    
    for txt_file in tqdm.tqdm(os.listdir(dataset_labels_root)):
        if not txt_file.endswith(".txt"):
            continue
        
        txt_full_path = os.path.join(dataset_labels_root, txt_file)
        slide_obj_label = read_and_filter_label_file(txt_full_path, filter_id=2)  # 2表示slide的类别
        slide_obj_label_coors = slide_obj_label[1:]  # 去除第一位的类别
        
        slide_obj_label_coors = [(1200*slide_obj_label_coors[i], 1920*slide_obj_label_coors[i + 1]) for i in range(0, len(slide_obj_label_coors), 2)]
        angle_gt = calculate_rotation_angle_box(slide_obj_label_coors) % 60
        
        
        if angle_gt > 10:
            angle_gt = angle_gt - 60
        elif angle_gt < -10:
            angle_gt = angle_gt + 60
        angle_gts_dict[txt_file] = angle_gt
    
    gt_save_full_path = os.path.join(dataset_labels_root, "angle_gts_by_slide.json")
    with open(gt_save_full_path, "w") as fw:
        json.dump(angle_gts_dict, fw, ensure_ascii = False, indent = 4)

def generate_angle_gt_by_slide_big_circle(dataset_labels_root: str) -> None:
    """通过slide和big_circle的坐标label信息生成数据集的angle标签。
    angle标签存放路径为dataset_labels_root/angle_gts_by_slide_big_circle.json。
    
    Args:
        dataset_labels_root: str, 数据集标签文件root目录。
    """
    angle_gts_dict = {}
    
    for txt_file in tqdm.tqdm(os.listdir(dataset_labels_root)):
        if not txt_file.endswith(".txt"):
            continue
        
        txt_full_path = os.path.join(dataset_labels_root, txt_file)
        slide_obj_label = read_and_filter_label_file(txt_full_path, filter_id=2)  # 2表示slide的类别
        slide_obj_label_coors = slide_obj_label[1:]  # 去除第一位的类别
        slide_obj_label_coors = [(1200*slide_obj_label_coors[i], 1920*slide_obj_label_coors[i + 1]) for i in range(0, len(slide_obj_label_coors), 2)]
        
        big_circle_obj_label = read_and_filter_label_file(txt_full_path, filter_id=0)  # 0表示big circle的类别
        big_circle_obj_label_coors = big_circle_obj_label[1:]  # 去除第一位的类别
        big_circle_obj_label_coors = [(1200 * big_circle_obj_label_coors[i], 1920 * big_circle_obj_label_coors[i + 1]) for i in range(0, len(big_circle_obj_label_coors), 2)]
        
        
        slide_center_x, slide_center_y = calculate_center(slide_obj_label_coors)
        big_circle_center_x, big_circle_center_y = calculate_center(big_circle_obj_label_coors)
        
        angle_gt = calculate_rotation_angle_line([(big_circle_center_x,
                                                                big_circle_center_y),
                                                                (slide_center_x,
                                                                slide_center_y)]) % 60
        
        if angle_gt > 10:
            angle_gt = angle_gt - 60
        elif angle_gt < -10:
            angle_gt = angle_gt + 60
        
        angle_gts_dict[txt_file] = angle_gt
    
    gt_save_full_path = os.path.join(dataset_labels_root, "angle_gts_by_slide_big_circle.json")
    with open(gt_save_full_path, "w") as fw:
        json.dump(angle_gts_dict, fw, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    if args.mode == "all":
        generate_angle_gt_by_plates(args.test_dataset_labels_root)
        generate_angle_gt_by_slide(args.test_dataset_labels_root)
        generate_angle_gt_by_slide_big_circle(args.test_dataset_labels_root)
    elif args.mode == "plates":
        generate_angle_gt_by_plates(args.test_dataset_labels_root)
    elif args.mode == "slide":
        generate_angle_gt_by_slide(args.test_dataset_labels_root)
    elif args.mode == "big_circle":
        generate_angle_gt_by_slide_big_circle(args.test_dataset_labels_root)
    # e.g. python 
    # python3 tools/generate_angle_gt_by_plates.py --test_dataset_labels_root ./../datasets/obb-project/labels/val