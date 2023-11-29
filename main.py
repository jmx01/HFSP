import copy
import multiprocessing
import operator
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.figure_factory import create_gantt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def Hungarian_Algorithm():
    """
    匈牙利算法，求解最优循环
    :return:最优循环列表
    """
    cost = dc1.values
    for i in range(cost.shape[0]):
        cost[i][i] = 10000
    row_ind, col_ind = linear_sum_assignment(cost)
    use_circle = []
    used_element = [0] * len(row_ind)
    while 0 in used_element:
        element = used_element.index(0)  # 可用的元素值
        used_element[element] = 1
        now = [element]
        ind = np.where(row_ind == now[-1])[0][0]
        while col_ind[ind] not in now:
            now.append(col_ind[ind])
            used_element[col_ind[ind]] = 1
            ind = np.where(row_ind == now[-1])[0][0]
        use_circle.append(now)
    return use_circle


def check_unique(p, part):
    """
    判断是否已经在种群中
    :param p: 子个体第一段
    :param part: 种群
    """
    if np.any(np.all(p != part, axis=1)):  # 如果新产生的子个体不在种群中则添加
        part = np.r_[part, [p]]
        fitness(p)
    return part


def add_initial_element(pop_num):
    """
    向种群添加不重复的精英解和随机解
    pop_num:总个体数
    return:空
    """
    part = np.array([copy.deepcopy(num)])  # 添加最初序列
    # 添加精英解
    out_layer = np.arange(len(circle_list))
    while len(part) <= population_elite_num:
        random.shuffle(out_layer)
        for k in range(len(circle_list)):
            start = random.randint(0, len(circle_list[k]) - 1)
            circle_list[k] = circle_list[k][start:] + circle_list[k][:start]
        part = check_unique(np.concatenate(circle_list), part)

    # 添加随机解
    while len(part) < pop_num:  # 当小于子个体数时
        random.shuffle(num)
        part = check_unique(num, part)


def fitness(p):
    """
    计算适应度
    :param p: 子个体，被解码的
    """
    delay_five = process(p, delay_initial, dp1, dc1, range(5))  # 前五个操作台结束时时间
    pop_six = buffer_now(delay_five, p, dp1, dc1)  # 某个体在5-6
    process_new(pop_six[0], pop_six[1], pop_six[2], dp1, dc1)  # 添加与计算


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

    for xx in range(len(now_list)):  # 第几个元素
        epoch = [delay_time[xx]]
        for j in arr:  # 每个元素在每台机器处的到达时间
            if xx == 0:  # 当是子个体的第一个加工工件
                epoch.append(dp.iat[j, now_list[xx]] + epoch[-1])  # 此工作台的延迟时间为操作时间+上一个工作台的延迟时间

            else:  # 当不是子个体的第一个加工工件
                start_time = max(end_time[-1][j - arr[0] + 1], epoch[-1]) + dc.iat[
                    now_list[xx - 1], now_list[xx]]  # 起始操作时间为max(上一个工件此操作台的延迟时间，此工件上一个工作台的操作时间)+换模时间
                epoch.append(start_time + dp.iat[j, now_list[xx]])  # 此工作台的延迟时间为起始操作时间+操作时间
        end_time.append(epoch)  # 每个工件在每个操作台的结束时间

    if not complete:
        return [era[-1] for era in end_time]
    else:
        return [era[-1] for era in end_time], end_time


def six_merge(delay1, delay2, index1, index2):
    """
    在六处加工结束后，生成出工序六的顺序
    :param delay1: 机器一延迟时间
    :param delay2: 机器二延迟时间
    :param index1: 顺序一
    :param index2: 顺序二
    :return:排好的顺序，对应的延迟时间
    """
    delay1, delay2 = np.array(delay1), np.array(delay2)
    index_buffer = np.hstack((index1, index2))
    delay_time = np.hstack((delay1, delay2))
    delay_arg = np.argsort(delay_time)
    index_sort = index_buffer[delay_arg]
    delay_sort = np.sort(delay_time)
    return index_sort, delay_sort


def buffer_now(delay: list, index: list, dp, dc):
    """
    计算进入buffer时的状态
    :param delay:出工序5时的延迟
    :param index:出工序5的顺序
    :param dp:操作时间
    :param dc:换模时间
    :return:进入buffer的顺序，延迟时间，1-buffer的顺序
    """
    first, second = [], []
    delay_time1, delay_time2 = [], []

    first.append(index[0])
    delay_time1.append(delay[0] + dp.iat[5, index[0]])
    deed = 1  # 已加工个数
    while deed < len(index):  # 当还有工件未完成时
        if random.random() < p:
            if not first:  # 当first为空时
                first.append(index[deed])
                delay_time1.append(delay[deed] + dp.iat[5, index[deed]])
            elif not second:  # first不为空,second为空
                second.append(index[deed])
                delay_time2.append(delay[deed] + dp.iat[5, index[deed]] * alpha)
            else:  # 均不为空
                if delay_time1[-1] > delay_time2[-1]:
                    second.append(index[deed])
                    delay_time2.append(
                        max(delay[deed], delay_time2[-1]) + dc.iat[second[-2], second[-1]] + dp.iat[
                            5, index[deed]] * alpha)
                else:
                    first.append(index[deed])
                    delay_time1.append(
                        max(delay[deed], delay_time1[-1]) + dc.iat[first[-2], first[-1]] + dp.iat[
                            5, index[deed]])
        else:
            if not second:  # 当second为空时
                second.append(index[deed])
                delay_time2.append(delay[deed] + dp.iat[5, index[deed]] * alpha)
            elif not first:  # second不为空,first为空
                first.append(index[deed])
                delay_time1.append(delay[deed] + dp.iat[5, index[deed]])
            else:  # 均不为空
                if delay_time1[-1] <= delay_time2[-1]:
                    first.append(index[deed])
                    delay_time1.append(
                        max(delay[deed], delay_time1[-1]) + dc.iat[first[-2], first[-1]] + dp.iat[
                            5, index[deed]])
                else:
                    second.append(index[deed])
                    delay_time2.append(
                        max(delay[deed], delay_time2[-1]) + dc.iat[second[-2], second[-1]] + dp.iat[
                            5, index[deed]] * alpha)
        deed += 1

    first, second = np.array(first), np.array(second)
    index_sort, delay_sort = six_merge(delay_time1, delay_time2, first, second)
    index_1t6 = [index, first, second]
    return index_sort, delay_sort, index_1t6


def update_inhibit_dict(index_all, delay, dp, dc):
    """
    更新禁忌表
    :param index_all:新的顺序
    :param delay: 进入7时的延迟时间
    :param dp: 操作时间
    :param dc: 换模时间
    :return: 空
    """
    ind = index_all[-1]
    name = [*index_all[0], -1]
    for ccc in range(len(index_all) - 1):
        name.extend(index_all[ccc + 1])
        name.append(-1)
    name = tuple(name)
    if name not in inhibit_dict.keys():
        inhibit_dict[name] = 1e5 / process(ind, delay, dp, dc)[-1]


def ox(solution1, solution2):
    """
    交换函数，交换两个等长列表的部分指定区域
    :param solution1: 列表1
    :param solution2: 列表2
    :return: [子列表1，子列表2]
    """
    "找两个不一样的点"
    if len(solution1) == 2:
        rand = [0, 1]
        random.shuffle(rand)
    else:
        rand = random.sample(range(0, len(solution1) - 1), 2)
    min_rand, max_rand = min(rand), max(rand)
    "生成不变区域"
    copy_mid = [solution1[min_rand:max_rand + 1], solution2[min_rand:max_rand + 1]]
    "生成改变区域"
    s1_head = solution1[:min_rand]
    s1_head.reverse()
    s1_tail = solution1[max_rand + 1:]
    s1_tail.reverse()
    s2_head = solution2[:min_rand]
    s2_head.reverse()
    s2_tail = solution2[max_rand + 1:]
    s2_tail.reverse()
    swap = [s2_head + s2_tail, s1_head + s1_tail]
    "生成子列表"
    c_new = []
    for ix in range(2):
        c_swap = []
        while swap[ix]:
            tmp = swap[ix].pop()
            if tmp not in copy_mid[ix]:
                c_swap.append(tmp)
        for c in copy_mid[1 - ix]:
            if c not in copy_mid[ix]:
                c_swap.append(c)
        c_new.append(c_swap[len(solution1) - max_rand - 1:] + copy_mid[ix] + c_swap[:len(solution1) - max_rand - 1])
    return c_new


def cross(index, key_words="cross", times=1):
    index_c = []
    index_copy = copy.deepcopy(index)
    for time in range(times):
        index1, index2 = np.random.choice(component_num, size=2, replace=False)
        if key_words == "cross":
            index_copy[-1][index2], index_copy[-1][index1] = index_copy[-1][index1], index_copy[-1][index2]
        elif key_words == "insert":
            value = index_copy[-1][index1]
            if index1 > index2:
                index_copy[-1] = np.delete(index[-1], index1)
                index_copy[-1] = np.insert(index_copy[-1], index2, value)
        elif key_words == "reverse":
            left, right = min(index1, index2), max(index1, index2)
            index_copy[-1][left:right] = index_copy[-1][left:right][::-1]
        else:
            pass
        index_c.append(copy.deepcopy(index_copy))
    return index_c


def neighborhood_search(index, delay, dp, dc, key_words="cross"):
    index_copy = cross(index, key_words, 10)
    for u in index_copy:
        update_inhibit_dict(u, delay, dp, dc)


def process_new(index, delay, index_16, dp, dc):
    """
    在7处的邻域搜索,求解
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
    update_inhibit_dict(index_origin, delay, dp, dc)
    # 交换一些顺序
    neighborhood_search(index_origin, delay, dp, dc, action[0])  # 交换
    neighborhood_search(index_origin, delay, dp, dc, action[1])  # 插入
    neighborhood_search(index_origin, delay, dp, dc, action[2])  # 倒置


def get_element_index(old_array, new_array):
    """
    用来寻找新数组在旧数组的索引
    :param old_array:旧数组
    :param new_array:新数组
    :return:索引数组
    """
    old_array = list(old_array)
    return np.array([old_array.index(era) for era in new_array])


def six_decode(now_list, delay_time, dp, dc, choose):
    """
    在六处的解码
    :param now_list:输入数组6-1或者6-2
    :param delay_time:输入对应延迟时间
    :param dp:操作时间
    :param dc:换模时间
    :param choose:选择机器1还是机器2
    :return:解码离开6时的延迟时间
    """
    delay_sort = []
    if choose == 1:
        for k in range(len(now_list)):
            if k == 0:
                delay_sort.append(delay_time[0] + dp.iat[5, now_list[0]])
            else:
                delay_sort.append(
                    max(delay_time[k], delay_sort[-1]) + dp.iat[5, now_list[k]] + dc.iat[now_list[k - 1], now_list[k]])
    else:
        for k in range(len(now_list)):
            if k == 0:
                delay_sort.append(delay_time[0] + dp.iat[5, now_list[0]] * alpha)
            else:
                delay_sort.append(max(delay_time[k], delay_sort[-1]) + dp.iat[5, now_list[k]] * alpha + dc.iat[
                    now_list[k - 1], now_list[k]])
    return delay_sort


def undo(p):
    """
    解开禁忌表key的元组，返回加工元组
    :param p:禁忌key，tuple
    :return:被解开的key，加工路径
    """
    p = np.array(p)
    indices = np.where(p == -1)[0]
    new_p = [[*p[0:component_num]]]
    for i in range(len(indices) - 1):
        new_p.append(p[indices[i] + 1:indices[i + 1]])
    return new_p


def decode_6(delay, end_time, po, pn):
    """
    返回查找的延迟时间和全部时间
    :param delay:延迟时间
    :param end_time:全部加工过程
    :param po: 原数组
    :param pn: 新数组
    :return:延迟，全部加工时间
    """
    index = get_element_index(po, pn)
    delay_6 = six_decode(pn, np.array(delay)[index], dp1, dc1, 1)  # 6-1 延迟时间
    for k in range(len(index)):
        end_time[index[k]].append(delay_6[k])
    return delay_6, end_time


def decode(p):
    """
    解码子个体
    :param p:某个子个体，格式为【1-5编码，6-1编码，6-2编码，7-10编码，适应度】
    :return:返回供甘特图使用的数据
    """
    delay_15, end_time_15 = process(p[0], delay_initial, dp1, dc1, range(5), complete=True)  # 1-5 延迟时间
    delay_61, end_time_15 = decode_6(delay_15, end_time_15, p[0], p[1])
    delay_62, end_time_15 = decode_6(delay_15, end_time_15, p[0], p[2])

    # 进入buffer的顺序和时间
    index_buffer, delay_buffer = six_merge(delay_61, delay_62, p[1], p[2])
    delay_buffer = delay_buffer[get_element_index(index_buffer, p[3])]

    # 7-10 延迟时间
    delay_710, end_time_710 = process(p[3], delay_buffer, dp1, dc1, complete=True)
    end_index = get_element_index(p[3], p[0])

    for e in range(len(end_time_15)):
        end_time_15[e].pop(0)
        end_time_710[end_index[e]].pop(0)
        end_time_15[e].extend(end_time_710[end_index[e]])

    start_time = []
    for out in range(len(end_time_15)):
        start_epoch = []
        for inside in range(len(end_time_15[out])):
            if p[0][out] in p[1] and inside == 5:
                start_epoch.append(end_time_15[out][inside] - dp1.iat[inside, p[0][out]] * alpha)
            else:
                start_epoch.append(end_time_15[out][inside] - dp1.iat[inside, p[0][out]])
        start_time.append(start_epoch)

    gantt = []
    datetime = pd.Timestamp('20231128 14:00:00')
    for component in range(component_num):
        for machine in range(10):
            gantt.append([p[0][component] + 1, datetime + pd.Timedelta(seconds=start_time[component][machine]),
                          datetime + pd.Timedelta(seconds=end_time_15[component][machine]), machine + 1])
    gantt = pd.DataFrame(gantt, columns=["Task", "Start", "Finish", "Resource"])
    gantt['Resource'] = gantt['Resource'].astype(str)
    fig = create_gantt(gantt, index_col='Resource', reverse_colors=True, show_colorbar=True, group_tasks=True)
    fig.show()
    print(p[0])
    print(p[1])
    print(p[2])
    print(p[3])
    print(delay_710[-1])


def population_to_batch():
    random.shuffle(population)
    batch_pop = [population[i:i + component_num] for i in range(0, len(population), component_num)]
    return batch_pop


def fetch_parents(bi):
    father_p, mother_p = undo(max(bi, key=lambda x: x[-1])[0])[0], undo(bi[random.randint(0, len(bi) - 1)][0])[0]
    return father_p, mother_p


def generate_child(f, m, bf, adorable_times=20, child_num=5):
    ad_num = 1e5 / bf * 0.6
    cf = [f, m]
    cdf = []
    for i in range(child_num):
        c1, c2 = ox(f, m)
        cf.append(c1)
        cf.append(c2)
    for cc in cf:
        cdf.append(process(cc, delay_initial, dp1, dc1, range(5)))

    while adorable_times >= 0:
        now = random.randint(0, len(cf) - 1)
        if cdf[now][-1] > ad_num:
            cf.pop(now)
            cdf.pop(now)
            new1, new2 = random.sample(cf, 2)
            c1, c2 = ox(copy.deepcopy(new1), copy.deepcopy(new2))
            cf.append(c1)
            cf.append(c2)
            cdf.append(process(c1, delay_initial, dp1, dc1, range(5)))
            cdf.append(process(c2, delay_initial, dp1, dc1, range(5)))
        adorable_times -= 1

    adorable_times = 20
    ad_num = 1e5 / bf * 0.8
    while len(cf) >= 2 and adorable_times > 0:
        now = random.randint(0, len(cf) - 1)
        if cdf[now][-1] > ad_num:
            cf.pop(now)
            cdf.pop(now)
        adorable_times -= 1

    for cc in range(len(cf)):
        index, delay, index1_6 = buffer_now(cdf[cc], cf[cc], dp1, dc1)
        process_new(index, delay, index1_6, dp1, dc1)


def work(bi):
    father, mother = fetch_parents(bi)
    generate_child(father, mother, population_best_one[-1])


# 零件号,机器号均减了1，记得最后要加1
path = 'D:/desk/industry_synthesis/数据/'
reed = 1
dp1 = pd.read_csv(path + 'case' + str(reed) + '_process.csv')
dc1 = pd.read_csv(path + 'case' + str(reed) + '_time.csv')

component_num = dp1.shape[1]  # 工件数
num = np.arange(component_num)  # 生成初始数据
circle_list = Hungarian_Algorithm()  # 最优循环
delay_initial = [0] * component_num  # 初始延迟时间，为工件数序列
population_elite_num = component_num  # 精英子个体数
action = ["cross", "insert", "reverse"]  # 邻域搜索的操作

out_circle = 10
population_random_num = 64  # 随机子个体数
population_num = population_elite_num + population_random_num  # 种群总个体数
half = population_num // 2
alpha = 1.2  # 六号工位二号机的加工时间系数
p = 0.9  # 天然选择一号机的概率

best_time_series = []
inhibit_dict = {}  # 空禁忌表

if __name__ == "__main__":
    add_initial_element(population_num)
    for kk in tqdm(range(out_circle), ncols=80, position=0, leave=True):
        inhibit_tuple = sorted(inhibit_dict.items(), key=operator.itemgetter(1), reverse=True)
        population = random.sample(inhibit_tuple[half:200], half)

        for k in range(half):
            population.append(inhibit_tuple[k])
        inhibit_dict = dict(inhibit_tuple)

        population_best_one = population[half + 1]
        batch = population_to_batch()

        for b in batch:

            # t1 = threading.Thread(target=work, args=(b,))
            # t2 = threading.Thread(target=work, args=(b,))
            # t3 = threading.Thread(target=work, args=(b,))
            # t4 = threading.Thread(target=work, args=(b,))
            # t5 = threading.Thread(target=work, args=(b,))
            # t6 = threading.Thread(target=work, args=(b,))

            # t1.start()
            # t2.start()
            # t3.start()
            # t4.start()
            # t5.start()
            # t6.start()

        inhibit_tuple = sorted(inhibit_dict.items(), key=operator.itemgetter(1), reverse=True)
        inhibit_dict = dict(inhibit_tuple)
        best_time_series.append(1e5 / inhibit_tuple[0][-1])
        print(inhibit_tuple[0])

    print(len(inhibit_dict))
    inhibit_tuple = sorted(inhibit_dict.items(), key=operator.itemgetter(1), reverse=True)
    decode(undo(inhibit_tuple[0][0]))
    plt.plot(best_time_series)
    plt.show()
    # 优秀解
    # decode([[5, 1, 6, 0, 4, 7, 3, 2], [6, 0, 4, 3], [5, 1, 7, 2], [5, 6, 0, 1, 7, 4, 3, 2]])
    # decode([[0, 6, 7, 5, 8, 3, 2, 1, 9, 4, 10, 15, 11, 12, 13, 14],
    #         [0, 7, 8, 2, 9, 10, 11, 13],
    #         [6, 5, 3, 1, 4, 15, 12, 14],
    #         [0, 6, 7, 5, 8, 3, 2, 1, 9, 4, 10, 15, 11, 12, 13, 14]])
