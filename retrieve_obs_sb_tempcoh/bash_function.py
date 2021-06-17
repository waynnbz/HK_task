import struct
import numpy as np
# 取消np.array默认的科学计数法显示
np.set_printoptions(suppress=True)

'''
这里记录了matlab常用的基础方法
'''


# 根据输入的格式要求，转换成对应的字节串长度
def get_data_type_size(types):
    if types in ['unsigned char', 'uchar', 'uint8']:
        size = 1
        format_type = '>B'
    elif types == 'char':
        size = 1
        format_type = '>c'
    elif types == ['int8', 'signed char', 'integer*1']:
        size = 1
        format_type = '>b'
    elif types in ['int16', 'integer*2', 'short']:
        size = 2
        format_type = '>h'
    elif types in ['uint16', 'unsigned int16', 'ushort', 'unsigned short']:
        size = 2
        format_type = '>H'
    elif types in ['int32', 'integer*4', 'int']:
        size = 4
        format_type = '>i'
    elif types in ['uint32',  'unsigned int', 'unsigned int32']:
        size = 4
        format_type = '>I'
    elif types == 'long':
        size = 4
        format_type = '>l'
    elif types in ['ulong', 'unsigned long']:
        size = 4
        format_type = '>L'
    elif types in ['float32',  'float']:
        size = 4
        format_type = '>f'
    elif types in ['int64', 'integer*8', 'long long']:
        size = 8
        format_type = '>q'
    elif types in ['uint64', 'unsigned int64', 'unsigned long long']:
        size = 8
        format_type = '>Q'
    elif types in ['float64', 'double']:
        size = 8
        format_type = '>d'
    else:
        print('input types error!')
        return None, None
    return size, format_type


'''
read_file 读取文件 将数据转换成对应的向量矩阵

输入参数:  
input_file 读取文件的路径 str (绝对路径 不能为空)
lines 输出文件的行数 int (不能为0 默认为1 不能被总数整除reshape会报错)
types 字节串转换格式 str (默认为float 如以 cpx开头结果转换成复数  详情见 get_data_type_size 函数)
line_start 从第几行开始截取数据 int（包括该行，非必要参数，不输入会从第一行开始）
line_end 截取到第几行 int（包括该行，非必要参数，不输入会取到文件最后一行）
column_start 从第几列开始截取数据 int（包括该行，非必要参数，不输入会从第一列开始）
column_end 截取到第几列 int（包括该行，非必要参数，不输入会取到文件最后一列）

输出参数:
data 处理过后的复数阵列 格式 np.array
count 文件元素总数 int
'''


def read_file(input_file, lines=1, types='float', line_start=None, line_end=None, column_start=None, column_end=None):
    in_file = open(input_file, 'rb').read()
    if types[0:3] == 'cpx':
        size, format_type = get_data_type_size(types[3:])
    else:
        size, format_type = get_data_type_size(types)
    if size and format_type:
        number = 0
        start = 0
        end = start + size
        file_long = len(in_file)
        real_part_list = []
        imaginary_part_list = []
        data_list = []
        while start < file_long:
            number += 1
            infile_bytes = in_file[start:end]
            format_bytes = struct.unpack(format_type, infile_bytes)[0]
            if types[0:3] == 'cpx':
                if number % 2 != 0:
                    real_part_list.append(format_bytes)
                else:
                    imaginary_part_list.append(format_bytes)
            else:
                data_list.append(format_bytes)
            start = end
            end = end + size
        if types[0:3] == 'cpx':
            plural_list = []
            for real, imag in zip(real_part_list, imaginary_part_list):
                plural = complex(real, imag)
                plural_list.append(plural)
            data_long = int(number/2)
            data_with = int(data_long/lines)
        else:
            data_long = number
            data_with = int(data_long / lines)
            plural_list = data_list
        plural_array = np.array(plural_list)
        plural_array = plural_array.reshape(lines, data_with)
        if line_start and line_end is None:
            plural_array = plural_array[line_start-1:]
        elif line_start is None and line_end:
            plural_array = plural_array[:line_end]
        elif line_start and line_end:
            plural_array = plural_array[line_start-1:line_end]
        if column_start and line_end is None:
            plural_array = plural_array[:, column_start-1:]
        elif column_start is None and line_end:
            plural_array = plural_array[:, :column_end]
        elif column_start and line_end:
            plural_array = plural_array[:, column_start-1:column_end]
        count = plural_array.size
        return plural_array, count
    else:
        print('parameter error!')
        return None, None


'''
write_file 该函数将处理过的数据输出成文件保存

输入参数:  
data 需要输出的数据 格式 np.array
outfile 输出的路径 str (绝对路径 不能为空)
types 字节串转换格式 str (默认为float 如以 cpx开头转换复数 详情见 get_data_type_size 函数)
'''


def write_file(data, outfile, types):
    if types[0:3] == 'cpx':
        size, format_type = get_data_type_size(types[3:])
    else:
        size, format_type = get_data_type_size(types)
    if size and format_type:
        out_file = open(outfile, 'wb')
        data_shape = data.shape
        data_list = []
        if types[0:3] == 'cpx':
            real_list = np.real(data.reshape(-1))
            imag_list = np.imag(data.reshape(-1))
            for real, imag in zip(real_list, imag_list):
                data_list.append(struct.pack(format_type, real))
                data_list.append(struct.pack(format_type, imag))
        else:
            data = data.reshape(-1)
            for i in data:
                data_list.append(struct.pack(format_type, i))
        out_file_array = np.array(data_list)
        if types[0:3] == 'cpx':
            out_file_array = out_file_array.reshape(int(np.prod(data_shape)), 2)
        else:
            out_file_array = out_file_array.reshape(int(np.prod(data_shape)/2), 2)
        out_file.write(out_file_array)
        out_file.close()
        return
