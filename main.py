import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 零件号,机器号均减了1，记得最后要加1
path = 'D:/desk/industry_synthesis/'
dp1 = pd.read_csv(path + 'case1_process.csv')  # case1_process
dc1 = pd.read_csv(path + 'case1_time.csv')  # case1_change
# dp2 = pd.read_csv(path+'case2_process.csv')  # case2_process
# dc2 = pd.read_csv(path+'case2_time.csv')  # case2_change

x = [8, 16]  # 零件数
alpha = 1.2  # 六号工位二号机的加工时间系数
num = np.arange(x[0])  # 生成初始数据
population_random_num = 64  # 随机子个体数
population_elite_num = x[0]  # 精英子个体数
population_num = population_elite_num + population_random_num  # 种群总个体数
inhibit_table = []  # 用来存储已经计算过的子个体
fitness_inhibit_table = []  # 用来存对应禁忌表的适应度


def add_element(pop_num):
    """
    向种群添加可重复精英解和不重复随机子个体
    pop_num:总个体数
    return:种群
    """
    part = np.array([copy.deepcopy(num)])  # 添加最初序列
    # 添加精英解

    # 添加随机解
    while len(part) < pop_num:  # 当小于子个体数时
        random.shuffle(num)
        if np.any(np.all(num != part, axis=1)):  # 如果新产生的子个体不在种群中则添加
            part = np.r_[part, [copy.deepcopy(num)]]
    return part


def process(now_list, delay_time, dp, dc, arr=range(6, 10), complete=False) -> list:
    """
    通过现在的加工序列，每个加工序列的延迟时间，生成对应操作台后的延迟时间
    :param now_list:加工序列
    :param delay_time:上一个部分的延迟时间
    :param dp:操作时间
    :param dc:换模时间
    :param arr:操作的工作台序号
    :param complete:判断是否完整显示
    :return:经过此部分操作后的延迟时间
    """
    end_time = []  # 每个操作台的结束时间

    for i in range(len(now_list)):  # 第几个元素
        epoch = [delay_time[i]]
        for j in arr:  # 每个元素在每台机器处的到达时间
            if i == 0:  # 当是子个体的第一个加工工件
                epoch.append(dp.iat[j, now_list[i]] + epoch[-1])  # 此工作台的延迟时间为操作时间+上一个工作台的延迟时间

            else:  # 当不是子个体的第一个加工工件
                start_time = max(end_time[-1][j - arr[0] + 1], epoch[-1]) + dc.iat[
                    now_list[i - 1], now_list[i]]  # 起始操作时间为max(上一个工件此操作台的延迟时间，此工件上一个工作台的操作时间)+换模时间
                epoch.append(start_time + dp.iat[j, now_list[i]])  # 此工作台的延迟时间为起始操作时间+操作时间
        end_time.append(epoch)  # 每个工件在每个操作台的结束时间

    if not complete:
        return [era[-1] for era in end_time]
    else:
        return [era[-1] for era in end_time], end_time


def buffer_now(delay: list, index: list, dp, dc, choose):
    """
    计算进入buffer时的状态
    :param delay:出工序5时的延迟
    :param index:出工序5的顺序
    :param dp:操作时间
    :param dc:换模时间
    :param choose:选择1号机还是2号机
    :return:进入buffer的顺序，延迟时间，1-buffer的顺序
    """
    first, second = [], []
    delay_time1, delay_time2 = [], []

    if choose == 1:  # 先走1号机
        first.append(index[0])
        delay_time1.append(delay[0] + dp.iat[5, index[0]])
    else:  # 先走2号机
        second.append(index[0])
        delay_time2.append(delay[0] + dp.iat[5, index[0]] * alpha)

    deed = 1  # 已加工个数
    while deed < len(index):  # 当还有工件未完成时
        if not first:  # 当first为空时
            first.append(index[deed])
            delay_time1.append(delay[deed] + dp.iat[5, index[deed]])
        elif not second:  # first不为空,second为空
            second.append(index[deed])
            delay_time2.append(delay[deed] + dp.iat[5, index[deed]] * alpha)
        else:  # 均不为空
            if delay_time1[-1] <= delay_time2[-1]:
                first.append(index[deed])
                delay_time1.append(
                    max(delay[deed], delay_time1[-1]) + dc.iat[index[deed - 1], index[deed]] + dp.iat[
                        5, index[deed]])
            else:
                second.append(index[deed])
                delay_time2.append(
                    max(delay[deed], delay_time2[-1]) + dc.iat[index[deed - 1], index[deed]] + dp.iat[
                        5, index[deed]] * alpha)
        deed += 1

    first, second = np.array(first), np.array(second)
    delay_time1, delay_time2 = np.array(delay_time1), np.array(delay_time2)
    index_buffer = np.hstack((first, second))
    delay_time = np.hstack((delay_time1, delay_time2))
    delay_arg = np.argsort(delay_time)
    index_sort = index_buffer[delay_arg]
    delay_sort = np.sort(delay_time)

    index_1t6 = [index, first, second]
    return index_sort, delay_sort, index_1t6


def five_to_six(delay: list, index: list, dp, dc):
    choose1 = buffer_now(delay, index, dp, dc, 1)  # 首次选择1号机
    choose2 = buffer_now(delay, index, dp, dc, 2)  # 首次选择2号机
    return choose1, choose2


def cross_index(index):
    index_new = copy.deepcopy(index)
    arg = range(len(index_new))
    return arg, index_new


def process_new(index, delay, index_16, dp, dc):
    """
    在7处的邻域搜索
    :param index:从buffer进入7的顺序
    :param delay:从buffer进入7的延迟时间
    :param index_16: 1-6的编码
    :param dp:操作时间
    :param dc:换模时间
    :return:此子个体较优的解
    """
    # first in first out
    index_origin = copy.deepcopy(index_16)
    index_origin.append(index)
    if index_origin not in inhibit_table:  # 当不在禁忌表中时
        inhibit_table.append(index_origin)  # 不再计算适应度
        fitness = 1e5 / process(index, delay, dp, dc)[-1]
        index_origin.append(fitness)  # 添加适应度，越大越好
        fitness_inhibit_table.append(fitness)
        print(index_origin)

    # 交换一些顺序
    arg1, index1 = cross_index(index)
    delay1 = process(index1, delay[arg1], dp, dc)  # 前五个操作台结束时时间
    return delay1[-1]


def six_decode(now_list, delay_time, dp, dc, choose):
    delay_sort = []
    if choose == 1:
        for k in range(len(now_list)):
            if k == 0:
                delay_sort.append(delay_time[0] + dp.iat[5, now_list[0]])
            else:
                delay_sort.append(delay_time[k] + dp.iat[5, now_list[k]] + dc.iat[now_list[k - 1], now_list[k]])
    else:
        for k in range(len(now_list)):
            if k == 0:
                delay_sort.append(delay_time[0] + dp.iat[5, now_list[0]] * alpha)
            else:
                delay_sort.append(delay_time[k] + dp.iat[5, now_list[k]] * alpha + dc.iat[now_list[k - 1], now_list[k]])
    return delay_sort


def decode(p):
    """
    解码子个体
    :param p:某个子个体，格式为【1-5编码，6-1编码，6-2编码，7-10编码，适应度】
    :return:返回一张甘特图
    """
    delay_15, end_time_15 = process(p[0], delay_initial, dp1, dc1, range(5), complete=True)
    print(delay_15)

    index_new = np.where(np.in1d(p[0], p[1]))[0]
    delay_61 = six_decode(p[1], np.array(delay_15)[index_new], dp1, dc1, 1)
    print(delay_61)

    index_new = np.where(np.in1d(p[0], p[2]))[0]
    delay_62 = six_decode(p[2], np.array(delay_15)[index_new], dp1, dc1, 2)
    print(delay_62)

    delay_710, end_time_710 = process(p[3], delay_initial, dp1, dc1, range(5), complete=True)


# 种群
population = add_element(population_num)  # 生成种群

for pop in population:
    # 某个体前五个操作台的时间
    delay_initial = [0] * x[0]  # 初始延迟时间，为工件数序列
    delay_five = process(pop, delay_initial, dp1, dc1, range(5))  # 前五个操作台结束时时间

    # 某个体在5-6时分化为两个个体
    pop_six1, pop_six2 = five_to_six(delay_five, pop, dp1, dc1)

    # 计算先走机器1时的适应度
    index_six, delay_six, index_one_to_six = pop_six1[0], pop_six1[1], pop_six1[2]
    process_new(index_six, delay_six, index_one_to_six, dp1, dc1)

    # 计算先走机器2时的适应度
    index_six, delay_six, index_one_to_six = pop_six2[0], pop_six2[1], pop_six2[2]
    process_new(index_six, delay_six, index_one_to_six, dp1, dc1)

decode(inhibit_table[0])
