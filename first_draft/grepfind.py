def grepfind(filename, pattern):
    '''
    逐行寻找目标参数名称，锁定其所在行，并返回对应数值
    *注意：只返还第一搜索结果
    :param filename: 目标文件夹
    :param pattern: 目标参数名称
    :return: 目标参数对应的数值
    '''

    with open(filename) as f:
        for line in f:
            if pattern.tolower() in line.tolower():
                return line.split(':')[1].strip()

    return None