import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon, MultiPoint


def get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])


def expand_polygon(polygon, scale=1.6):
    """
    将四边形外扩scale倍
    polygon: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    scale: 外扩倍数
    """
    poly = Polygon(polygon)
    distance = poly.area * scale / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        expanded_coords = np.array(offset.Execute(distance), dtype=np.int32)
        points, sside = get_mini_boxes(expanded_coords)
    except:
        centroid = poly.centroid
        expanded_coords = []
        for x, y in polygon:
            new_x = centroid.x + 1.5 * (x - centroid.x)
            new_y = centroid.y + 1.5 * (y - centroid.y)
            expanded_coords.append((new_x, new_y))
        return expanded_coords

    return points


def is_intersect(polygon1, polygon2):
    """
    检查两个四边形是否相交
    polygon: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    return poly1.intersects(poly2)

def cluster_polygons(polygons):
    """
    对四边形进行聚类
    polygons: list of [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    expanded_polygons = [expand_polygon(polygon) for polygon in polygons]
    clusters = []
    visited = [False] * len(polygons)
    
    for i in range(len(polygons)):
        if not visited[i]:
            cluster = []
            stack = [i]
            while stack:
                idx = stack.pop()
                if not visited[idx]:
                    visited[idx] = True
                    cluster.append(polygons[idx])
                    for j in range(len(polygons)):
                        if not visited[j] and is_intersect(expanded_polygons[idx], expanded_polygons[j]):
                            stack.append(j)
            clusters.append(cluster)
    
    return clusters

def minimum_bounding_rectangle(cluster):
    """
    计算聚类后的最小外接矩形
    cluster: list of [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    points = [point for polygon in cluster for point in polygon]
    multi_point = MultiPoint(points)
    # min_rect = multi_point.minimum_rotated_rectangle
    # bbox = np.array(list(min_rect.exterior.coords)[:-1], dtype=np.int32).reshape(4, 2)

    # 获取水平外接矩形的边界框
    min_x, min_y, max_x, max_y = multi_point.bounds
    
    # 构造边界框的四个顶点
    bbox = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ], dtype=np.int32)
    
    # 计算面积
    area = (max_x - min_x) * (max_y - min_y)

    return bbox, area