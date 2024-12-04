import math

def convert_pixel_to_cartesian(coors: list) -> list:
    """图片上的坐标，y方向的大小与笛卡尔坐标系是反着的，该函数用于调整这种相对大小关系。
    
    Args:
        coors: list, 像素坐标系中的坐标[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Return:
        list, （仅逻辑上）笛卡尔坐标系的坐标
    """
    converted_coors = [(x, -y) for (x, y) in coors]
    return converted_coors
    

def calculate_rotation_angle_box(coors: list) -> float:
    """通过box的四个顶点，计算box的旋转角度(逆时针为正)
    
    Args:
        coors: list, box的四个顶点的坐标（像素坐标系）[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Return:
        float, box的旋转角度
    """
    assert len(coors) == 4
    
    coors = convert_pixel_to_cartesian(coors)
    x1, y1, x2, y2, x3, y3 = coors[0][0], coors[0][1], coors[1][0], coors[1][1], coors[2][0], coors[2][1]
   
    edg1_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    edg2_len = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    
    start_x = x1
    start_y = y1
    end_x = x2
    end_y = y2
    
    if edg1_len < edg2_len:
        start_x = x2
        start_y = y2
        end_x = x3
        end_y = y3    
    
    
    angle_deg = calculate_rotation_angle(start_x, start_y, end_x, end_y)

    if angle_deg % 90 == 0:
        angle_deg = 0

    return angle_deg

def calculate_rotation_angle_line(coors):
    """通过line的两个顶点，计算line的旋转角度(逆时针为正)
    
    Args:
        coors: list, line的两个顶点的坐标（像素坐标系）[(x1, y1), (x2, y2)]
    
    Return:
        float, line的旋转角度
    """
    assert len(coors) == 2
    coors = convert_pixel_to_cartesian(coors)
    
    start_x, start_y, end_x, end_y = coors[0][0], coors[0][1], coors[1][0], coors[1][1]
    
    angle_deg = calculate_rotation_angle(start_x, start_y, end_x, end_y)
    
    if angle_deg % 90 == 0:
        angle_deg = 0

    return angle_deg
    
    

def calculate_rotation_angle(start_x, start_y, end_x, end_y):
    # 计算两个点之间的向量
    dx = end_x - start_x
    dy = end_y - start_y

    angle_rad = math.atan(dy/dx)

    # 将弧度转换为角度
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_center(vertices: list)-> tuple:
    """
    计算旋转矩形框的中心点坐标

    Args:
        vertices, list 四个顶点坐标的列表，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    Return:
        中心点坐标 (cx, cy)
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]

    # 计算中心点坐标
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4

    return (cx, cy)
    