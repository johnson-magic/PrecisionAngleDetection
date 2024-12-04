import os
import sys
import json
import tqdm
import shutil
import argparse
from ultralytics import YOLO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.angle_utils import calculate_rotation_angle_box, \
    calculate_rotation_angle_line, calculate_center
from utils.path_utils import make_metric_about_save_path
from utils.plot_utils import draw_rotated_rectangle, draw_label, draw_axes, plot_scatter_with_stats

# def top_n_largest_values(data_dict, n=5):
#     # 使用 sorted() 函数和 lambda 表达式按值排序
#     top_n = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)[:n]
#     return top_n  # 返回一个列表，包含前 n 个最大值的键值对

# top_5 = top_n_largest_values(diff_0_dict)
# print(top_5)

def add_angle_result(result: list) -> list:
    """在已有detection的基础上，增添angle结果
    [
    {
        "name": "big_circle",
        "class": 0,
        "confidence": 0.98939,
        "box": {
            "x1": 89.43424,
            "y1": 613.41119,
            "x2": 91.44656,
            "y2": 1547.61963,
            "x3": 1026.3363,
            "y3": 1545.60596,
            "x4": 1024.3241,
            "y4": 611.39752
        }
    },
    {
        "name": "slide",
        "class": 2,
        "confidence": 0.96916,
        "box": {
            "x1": 596.28107,
            "y1": 840.74811,
            "x2": 743.41351,
            "y2": 933.38007,
            "x3": 860.48712,
            "y3": 747.4256,
            "x4": 713.35468,
            "y4": 654.79364
        }
    },
    {
        "name": "plates",
        "class": 1,
        "confidence": 0.96092,
        "box": {
            "x1": 104.76064,
            "y1": 963.10791,
            "x2": 105.705,
            "y2": 1208.11426,
            "x3": 1015.79028,
            "y3": 1204.60645,
            "x4": 1014.84595,
            "y4": 959.6001
        }
    }
]
    """
    
    # 第一步，需要做一个过滤，过滤掉误检的detection结果
    result = [x for x in result if x["confidence"] > 0.6]

    plates_res = [x for x in result if x["name"] == "plates"]
    assert len(plates_res) == 1
    plates_res = plates_res[0]  # 清爽一点
    
    slide_res = [x for x in result if x["name"] == "slide"]
    assert len(slide_res) == 1
    slide_res = slide_res[0]
    
    big_circle_res= [x for x in result if x["name"] == "big_circle"]
    assert len(big_circle_res) == 1
    big_circle_res = big_circle_res[0]
    
    # 以方案1的方式添加angle（基于plates坐标）
    plates_coors = [(plates_res["box"][f"x{i}"], plates_res["box"][f"y{i}"])
                       for i in range(1, 5)]
    plates_res["angle"] = calculate_rotation_angle_box(plates_coors)
    
    # 以方案2的方式添加angle (基于slide坐标)
    slide_coors = [(slide_res["box"][f"x{i}"], slide_res["box"][f"y{i}"])
                       for i in range(1, 5)]
    slide_res["angle"] = calculate_rotation_angle_box(slide_coors) % 60
    if slide_res["angle"] > 10:
        slide_res["angle"] = slide_res["angle"] - 60
    elif slide_res["angle"] < -10:
        slide_res["angle"] = slide_res["angle"] + 60
    
    
    # 以方案3的方式添加angle(基于big_circle && slide坐标)
    big_circle_coor = [(big_circle_res["box"][f"x{i}"], big_circle_res["box"][f"y{i}"])
                       for i in range(1, 5)]
    big_circle_center_x, big_circle_center_y = calculate_center(big_circle_coor)
    slide_center_x, slide_center_y = calculate_center(slide_coors)
    big_circle_res["angle"] = calculate_rotation_angle_line([(big_circle_center_x,
                                                               big_circle_center_y),
                                                               (slide_center_x,
                                                               slide_center_y)]) % 60
    if big_circle_res["angle"] > 10:
        big_circle_res["angle"] = big_circle_res["angle"] - 60
    elif big_circle_res["angle"] < -10:
        big_circle_res["angle"] = big_circle_res["angle"] + 60
    
    result = []
    result.append(plates_res)
    result.append(slide_res)
    result.append(big_circle_res)
    
    return result 
    
    
    

def inference(best_model_path: str, test_dataset_images_root: str,
              inference_save_path: str, vis_save_path: str) -> None:
    """模型推理并保存推理结果、可视化结果
    
    Args:
        best_model_path: str, 待推理（最优）模型路径
        test_dataset_images_root： str, 测试图片路径
        inference_save_path： str, 推理结果保存路径
        vis_save_path：str, 可视化结果保存路径   
    """

    # Load a model
    model = YOLO(best_model_path)
    
    # inference
    # generate obj pred for val testdataset
    for jpg in tqdm.tqdm(os.listdir(test_dataset_images_root)):
        # Predict with the model
        
        # detection结果
        src_path = os.path.join(test_dataset_images_root, jpg)
        results = model(src_path) # predict on an image
        json_result = results[0].to_json()  # str
        json_result = json.loads(json_result)  # json obj
        
        # angle结果（目前支持三种形式）
        json_result = add_angle_result(json_result)
        
        # 保存推理结果
        dst_path = os.path.join(inference_save_path, jpg[:-4] + ".json")
        with open(dst_path, "w") as fw:
            json.dump(json_result, fw, ensure_ascii = False, indent = 4)
        
        # 保存可视化结果
        dst_path = os.path.join(vis_save_path, jpg)
        shutil.copyfile(src_path, dst_path)
        
        for res in json_result:
            if res["name"] == "plates":
                # 绘制预测框，和角度
                angle = res["angle"]
                coor = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                       for i in range(1, 5)]
                draw_rotated_rectangle(dst_path, coor, (11, 219, 235), dst_path)
                label = res["name"] + " " + str(round(res["confidence"], 2)) + " " + str(round(angle, 2)) 
                draw_label(dst_path, coor, label, (11, 219, 235), dst_path)
                draw_axes(dst_path, coor[2], dst_path)
                       
            elif res["name"] == "slide":
                 # 绘制预测框，和角度
                angle = res["angle"]
                coor = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                       for i in range(1, 5)]
                draw_rotated_rectangle(dst_path, coor, (243, 243, 243), dst_path)
                label = res["name"] + " " + str(round(res["confidence"], 2)) + " " + str(round(angle, 2)) 
                draw_label(dst_path, coor, label, (243, 243, 243), dst_path)
                draw_axes(dst_path, coor[2], dst_path)
                
            elif res["name"] == "big_circle":
                # 绘制预测框，和角度
                angle = res["angle"]
                coor = [(res["box"][f"x{i}"], res["box"][f"y{i}"])
                       for i in range(1, 5)]
                draw_rotated_rectangle(dst_path, coor, (4, 42, 255), dst_path)
                label = res["name"] + " " + str(round(res["confidence"], 2)) + " " + str(round(angle, 2)) 
                draw_label(dst_path, coor, label, (4, 42, 255), dst_path)
                draw_axes(dst_path, coor[2], dst_path)

def compute_diff(test_dataset_labels_root, inference_save_path, badcase_save_path, vis_save_path, metric_save_path):
    # badcase and metric
    
    # badcase的保存路径分为3个子文件夹，方案1， 方案2， 方案3
    plates_badcase_save_path = os.path.join(badcase_save_path, "方案1")
    os.makedirs(plates_badcase_save_path)
    slide_badcase_save_path = os.path.join(badcase_save_path, "方案2")
    os.makedirs(slide_badcase_save_path)
    big_circle_badcase_save_path = os.path.join(badcase_save_path, "方案3")
    os.makedirs(big_circle_badcase_save_path)
    
    with open(os.path.join(test_dataset_labels_root, "angle_gts.json"), "r") as fr:
        angle_gts = json.load(fr)
        
    diff_plates_dic = {}
    diff_slide_dic = {}
    diff_big_circle_dic = {}

    for k, v in angle_gts.items():
        with open(os.path.join(inference_save_path, k[:-4] + ".json")) as fr:
            preds = json.load(fr)
        
        for pred in preds:
            if pred["name"] == "plates":
                angle_pred = pred["angle"]
                diff_plates_dic[k] = abs(v - angle_pred)
                coor = [(pred["box"][f"x{i}"], pred["box"][f"y{i}"])
                       for i in range(1, 5)]
                coor = coor[1:] + [coor[0]]  # 绘制在第二个点
                if abs(v - angle_pred) != 0:
                    draw_label(os.path.join(vis_save_path, k[:-4] + ".jpg"), coor, str(round(v, 2)) , (11, 219, 235), os.path.join(plates_badcase_save_path, k[:-4] + ".jpg"))
                    
            elif pred["name"] == "slide":
                angle_pred = pred["angle"]
                diff_slide_dic[k] = abs(v - angle_pred)
                coor = [(pred["box"][f"x{i}"], pred["box"][f"y{i}"])
                       for i in range(1, 5)]
                coor = coor[1:] + [coor[0]]  # 绘制在第二个点
                if abs(v - angle_pred) != 0:
                    draw_label(os.path.join(vis_save_path, k[:-4] + ".jpg"), coor, str(round(v, 2)) , (243, 243, 243), os.path.join(slide_badcase_save_path, k[:-4] + ".jpg"))
                    
            elif pred["name"] == "big_circle":
                angle_pred = pred["angle"]
                diff_big_circle_dic[k] = abs(v - angle_pred)
                coor = [(pred["box"][f"x{i}"], pred["box"][f"y{i}"])
                       for i in range(1, 5)]
                coor = coor[1:] + [coor[0]]  # 绘制在第二个点
                if abs(v - angle_pred) != 0:
                    draw_label(os.path.join(vis_save_path, k[:-4] + ".jpg"), coor, str(round(v, 2)) , (4, 42, 255), os.path.join(big_circle_badcase_save_path, k[:-4] + ".jpg"))
            
    plot_scatter_with_stats(list(diff_plates_dic.values()), os.path.join(metric_save_path, "方案1.png"))
    plot_scatter_with_stats(list(diff_slide_dic.values()), os.path.join(metric_save_path, "方案2.png"))
    plot_scatter_with_stats(list(diff_big_circle_dic.values()), os.path.join(metric_save_path, "方案3.png"))
    

def metric(test_dataset_images_root: str, test_dataset_labels_root: str, best_model_path: str)-> None:
    """度量模型在测试集上的效果。包含以下功能：
    * 推理结果（包含目标检测、角度检测）
    * 可视化结果（包含目标检测、角度检测）
    * badcase结果（角度检测）
    * 度量结果（包含分布图、文本指标）
    
    Args:
        test_dataset_images_root: str, 测试图片的root目录
        test_dataset_labels_root: str, 测试图片标签的root目录
        best_model_path: str, 最优模型文件
    """
    save_root_path = os.path.dirname(best_model_path)  # 将度量结果存放于模型目录同文件夹
    inference_save_path, vis_save_path, badcase_save_path, metric_save_path = \
        make_metric_about_save_path(save_root_path)
    
    inference(best_model_path, test_dataset_images_root, inference_save_path, vis_save_path)
    compute_diff(test_dataset_labels_root, inference_save_path, badcase_save_path, vis_save_path, metric_save_path)
    



def arg_parser():
    parser = argparse.ArgumentParser(description="度量模型质量")
    
    # 添加参数
    parser.add_argument('--test_dataset_images_root', type=str, required=True, help='测试集图片路径')
    parser.add_argument('--test_dataset_labels_root', type=str, required=True, help='测试集真值路径')
    parser.add_argument('--best_model_path', type=str, required=True, help='测试（最好）模型路径')

    return parser

if __name__ == "__main__":
    parser = arg_parser()
    # test_dataset_images_root = "datasets/obb-project/images/val/"
    # test_dataset_labels_root = "datasets/obb-project/labels/val/"
    # best_model_path = "runs/obb/train/weights/best.pt"
    args = parser.parse_args()
    
    metric(args.test_dataset_images_root, args.test_dataset_labels_root, args.best_model_path)
    
    
    