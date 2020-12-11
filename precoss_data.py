# 需要导入的库，struct是一个很常用的二进制解析库
import numpy as np
import struct


def decode_idx3_ubyte(idx3_ubyte_file):  # 此函数用来解析idx3文件，idx3_ubyte_filec指定图像文件路径
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offest = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offest)
    print('魔数：%d,图片数量：%d，图片大小：%d%d' % (magic_number, num_images, num_rows, num_cols))
    # 解析数据集
    image_size = num_rows * num_cols
    offest += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析%d' % (i + 1) + '张')
            images[i] = np.array(
                struct.unpack_from(fmt_image, bin_data, offest)).reshape((num_rows, num_cols))
    offest += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):  # 解析idx1文件函数，idx1_ubyte_file指定标签文件路径
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offest = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offest)
    print('魔数：%d，图片数量：%d张' % (magic_number, num_images))
    # 解析数据集
    offest += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析：%d' % (i + 1) + '张')
    labels[i] = struct.unpack_from(fmt_image, bin_data, offest)[0]
    offest += struct.calcsize(fmt_image)
    print(labels[0])
    return labels

file1='train-images.idx3-ubyte'
file2='train-labels.idx1-ubyte'

a=decode_idx3_ubyte(file1)
a1=decode_idx1_ubyte(file2)
a3=1
