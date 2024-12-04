import os

def make_subdir(root: str, subdirname: str) -> str:
    """make并返回目录
    Args:
        root: str, 根目录
        subdirname: str, 子文件夹名称
    
    Return:
        subdir路径
    """
    subdir_path = os.path.join(root, subdirname)
    os.mkdir(subdir_path)
    
    return subdir_path

def make_metric_about_save_path(root: str)-> tuple:
    """生成度量相关的文件夹
    
    Args:
        root: str, 度量相关信息保存的root目录
    
    * inference_save_path
    * vis_save_path
    * badcase_save_path
    * metric_save_path
    """
    inference_save_path = make_subdir(root, "inference")
    vis_save_path = make_subdir(root, "vis")
    badcase_save_path = make_subdir(root, "badcase")
    metric_save_path = make_subdir(root, "metric")
    
    return inference_save_path, vis_save_path, badcase_save_path, metric_save_path