import sys
sys.path.insert(0, './deps')

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
import math
from multiprocessing import Pool

def split_image(image_path, output_folder, tile_size=(256, 256)):
    # 打开大图
    img = Image.open(image_path)
    img_width, img_height = img.size

    # 获取图片的文件名（不带扩展名）
    image_name = os.path.basename(image_path).split('.')[0]

    # 计算大图的网格尺寸
    tiles_x = img_width // tile_size[0]
    tiles_y = img_height // tile_size[1]

    # 切分并保存小图
    for i in range(tiles_x):
        for j in range(tiles_y):
            # 定义每个小图的区域 (左, 上, 右, 下)
            left = i * tile_size[0]
            top = j * tile_size[1]
            right = (i + 1) * tile_size[0]
            bottom = (j + 1) * tile_size[1]

            # 裁剪小图
            tile = img.crop((left, top, right, bottom))

            # 根据大图的名字和切片的位置命名小图
            tile_name = f"{image_name}_tile_{i}_{j}.tif"
            tile.save(os.path.join(output_folder, tile_name))

    print(f"Image '{image_name}' split into {tiles_x * tiles_y} tiles and saved in '{output_folder}'.")

def split_images_in_folder(input_folder, output_folder, tile_size=(20, 20)):
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            # 构造文件的完整路径
            image_path = os.path.join(input_folder, filename) # 转换为NumPy数组

            split_image(image_path, output_folder, tile_size)

# 示例使用
image_path = "00DATASET/NBTwithLabel"  # 大图路径
output_folder = "00DATASET/mini"         # 保存小图的文件夹
split_images_in_folder(image_path, output_folder, tile_size=(50,50))

from PIL import Image
import os

def reduce_image_resolution(image_path, output_path, scale_factor=0.5):
    """
    将图片的像素降低，按比例缩放。

    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    :param scale_factor: 缩放比例（默认为0.5，将图片缩小到原始大小的一半）
    """
    # 打开图片
    img = Image.open(image_path)
    
    # 获取原始尺寸
    original_width, original_height = img.size
    
    # 计算缩小后的尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 调整图片尺寸
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # 保存缩小后的图片
    resized_img.save(output_path)
    print(f"Image saved at {output_path} with resolution {new_width}x{new_height}.")

def reduce_resolution_in_folder(input_folder, output_folder, scale_factor=0.5):
    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('tif'))]

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历并处理每张图片
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        reduce_image_resolution(input_path, output_path, scale_factor)

# 示例使用
input_folder = "00DATASET/test"   # 输入文件夹路径
output_folder = "00DATASET/test05" # 输出文件夹路径
reduce_resolution_in_folder(input_folder, output_folder, scale_factor=0.7)


# 将灰度值<160的点转换为点云
def img2points(image):
    threshold = 120
    points = []
    
    # 遍历每个像素点，转换为点云坐标
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < threshold:
                points.append([i / image.shape[0], j / image.shape[1]])  # 归一化处理

    points = np.array(points)
    return points

# 打印并显示点云
def plotpoints(points):
    plt.scatter(points[:, 1], -points[:, 0], s=1)  # 颠倒y轴以匹配图像坐标系
    plt.title("Point Cloud from Image")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# 读取指定文件夹中的所有TIF图像，并将其处理为点云
def process_tif_images(image_folder):
    labels = []
    all_point_clouds = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        if filename.endswith(".tif"):
            # 构造文件的完整路径
            image_path = os.path.join(image_folder, filename)
            
            # 打开TIF图像并转换为灰度图
            image = Image.open(image_path).convert('L')  # 将图像转换为灰度
            image_np = np.array(image)  # 转换为NumPy数组

            # 将图像转换为点云
            points = img2points(image_np)
            all_point_clouds.append(points)

            # 打印处理后的点云信息并绘图
            print(f"Processed {filename}, Number of points: {len(points)}")
            plotpoints(points)

    return all_point_clouds

# 设置TIF图像的文件夹路径
image_folder = "00DATASET/test05"  
# 替换为你的TIF数据集文件夹路径

# 处理TIF数据集并添加标签
point_clouds = process_tif_images(image_folder)

# 输出所有处理后的点云和标签
print(f"Processed {len(point_clouds)} images.")

point_clouds[2].shape

# 5 * 102
# 3 * 112
# 3 * 20
# 4 * 36
# y = [102]*5 + [112]*3 + [20]*3 + [36]*4
# 500 * 15
# 3 min
# 102 & 112 相似，20 & 36 相似

dgms = []
j = 0
for i in point_clouds:
    st = gd.RipsComplex(points=i, max_edge_length=1.).create_simplex_tree(max_dimension=2)
    dgms.append(st.persistence())
    plot = gd.plot_persistence_diagram(dgms[j])
    j = j + 1

# persistence diagram of random point
def Printpd(sam):
    st = gd.RipsComplex(points=sam, max_edge_length=1.).create_simplex_tree(max_dimension=2)
    dgm = st.persistence()
    plot = gd.plot_persistence_diagram(dgm)

Printpd(point_clouds[0])

# extrct birth & death
bds = []
for dgm in dgms:
    arr = [(birth, 1) if math.isinf(death) else (birth, death) for _, (birth, death) in dgm]
    # merged_arr = merge_close_tuples(arr, 0.0001) 
    bds.append(arr)
# print(bds[1])

import numpy as np
from sklearn import svm, datasets
from matplotlib.pylab import plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from multiprocessing import Pool

X = bds
y =  y = [102]*5 + [112]*3 + [20]*4 + [36]*3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
clf = svm.SVC(kernel='precomputed')

# kernel_taylor
def taylor_ex(x, terms=4):
    result = 1.0  # 初始项，即泰勒展开的第一项
    factorial = 1.0
    power = x

    for i in range(1, terms):
        factorial *= i
        result += power / factorial
        power *= x

    return result

def k_sigma_p(F, G, sigma=0.1): # distance between two diagram
    # distance between each point p & q
    sum1 = 0
    sum2 = 0
    for p in F:
        for q in G:
            norm_sq = np.linalg.norm(np.array(p) - np.array(q)) ** 2
            # if(norm_sq > 1.2): print(norm_sq)
            sum1 += taylor_ex(-norm_sq / (8 * sigma))
            norm_sq_res = np.linalg.norm(np.array(p) - np.array(q[::-1])) ** 2
            sum2 += taylor_ex(-norm_sq_res / (8 * sigma))

    return (sum1 - sum2)/ (8 * sigma * math.pi)

def compute_k_sigma_p_pair(pair):
    F, G = pair
    return k_sigma_p(F, G)


def my_k_matrix_p(A, B):
    with Pool() as pool:
        # 创建所有要计算的参数对
        pairs = [(a, b) for a in A for b in B]
        # 并行计算每个参数对的结果
        results = pool.map(compute_k_sigma_p_pair, pairs)
    
    # 将结果转换为矩阵
    k = np.array(results).reshape(len(A), len(B))
    return k
'''
def my_k_matrix_p(A,B):
    k = []
    for a in A:
        line = []
        for b in B:
            line.append(k_sigma_p(a,b))
        k.append(line)
    return k
'''

# k = np.array(my_k_matrix_p(bds10,bds10))
# print(k)

kernel = my_k_matrix_p(X_train, X_train)
kernel_test = my_k_matrix_p(X_test, X_train)

clf.fit(kernel, y_train)
y_pred = clf.predict(kernel_test)
acc = accuracy_score(y_test, y_pred)
print(y_pred)
print(y_test)
print("Acc: ", acc)


