"""
    @Author: sfwy
    @Description: 数据加载器进行处理

"""
import numpy as np
import pandas as pd


class DataLoader:

    def __init__(self):

        self.street_dict = {}  # 每个小段link字典 从0开始
        pass

    def init(self, link_path, weather_path):

        # 初始化每天的天气以及其他的
        self.weather_dict = {'cloudy': 0, 'heavy rain': 1, 'moderate rain': 2, 'rainstorm': 3, 'showers': 4,
                             0: 'cloudy', 1: 'heavy rain', 2: 'moderate rain', 3: 'rainstorm', 4: 'showers'}

        # 星期几
        self.week_dict = {  # 余 -> 星期
            3: 1, 4: 2, 5: 3, 6: 4, 0: 5, 1: 6, 2: 0
        }

        # link_id

        self.link_dict = {}

        with open(link_path, "r") as file:
            for line in file.readlines():
                link_id = int(line.split(" ")[0])
                self.link_dict[link_id] = len(self.link_dict)

        self.day_weather = {}

        day_weathers = np.array(pd.read_csv(weather_path))
        for weather in day_weathers:
            self.day_weather[str(weather[0])] = self.weather_dict[str(weather[1])]

    def description(self):
        head_s = {
            "order": "string (int) 订单id",
            "ata": "float 真实行程总时间",
            "distance": "float 路线距离",
            "simple eta": "float 出发时刻平均通行时间加权",
            "driver id": "int 司机id",
            "slice id": "int 出发时刻时间片id(每5分钟一个)"
        }
        # 多个link 空格隔开，内部逗号隔开
        link_s = {
            "link id": "int 道路小段id",
            "link time": "float 出发时刻道路小段的平均通行时间",
            "link ratio": "float 道路小段通行比例",
            "link current status": "int 出发时刻道路小段的路况状态 0,1,2,3",
            "link arrival status": "int 到达时刻道路小段的路况状态"
        }
        # 空格隔开
        cross_s = {
            "cross id": "int 红绿灯路口id",  # 开始路id-> 结束路id
            "cross time": "float 挖掘路口通行时间"
        }

    def processStr(self, line, weather=None, week=None):

        strs = line.split(";;")
        # head
        head_s = strs[0].split(" ")
        order, ata, distance, simple_ata, driver_id, slice_id = head_s[0], float(head_s[1]), float(head_s[2]), float(
            head_s[3]), int(head_s[4]), int(head_s[5])

        ata = (ata - 11.) / (11747. - 11)
        distance = (distance - 11.6293) / (144247.274 - 11.6293)
        simple_ata = (simple_ata - 2.0)/ (11161.0 - 2.0)
        driver_id = (driver_id - 0) / (80886. - 0)
        slice_id = (slice_id - 0) / (287. - 0)

        link_s = strs[1].split(" ")

        X_trains = []
        X_order = []

        X_cross = []

        # 每个link 记录一个
        for link in link_s:
            link_id, lin = link.split(":")
            lin = lin.split(",")

            link_id = int(link_id)
            X_order.append(link_id)
            try:
                link_id = self.link_dict[link_id] / 882389.
            except KeyError:
                return [], [], 0, simple_ata, order

            link_time = float(lin[0]) / 10.
            link_ratio = float(lin[1])
            #link_current_status = int(lin[2])
            link_current_status = [0, 0, 0, 0]
            link_current_status[int(lin[2]) if int(lin[2])<4 else 3] = 1

            #link_arrival_status = int(lin[3])
            link_arrival_status = [0, 0, 0, 0]

            link_arrival_status[int(lin[3]) if int(lin[3])<4 else 3] = 1

            X_trains.append(weather + week + [distance, simple_ata, driver_id, slice_id, link_id, link_time, link_ratio] +
                             link_current_status + link_arrival_status)

            # time += link_time
            # 添加有效的小段路数据

        if (strs[2].strip() == ""):
            return np.array(X_trains, np.float32), np.array(X_cross, np.float32), ata, simple_ata, order
        # print(X_order)
        cross_s = strs[2].split(" ")

        for cross in cross_s:
            cross_id, cross_time = cross.split(":")

            start_link_id, end_link_id = cross_id.split("_")
            start_link_id = int(start_link_id)
            end_link_id = int(end_link_id)
            cross_time = float(cross_time)

            try:
                start_link = X_trains[X_order.index(start_link_id)]
                end_link = X_trains[X_order.index(end_link_id)]
            except ValueError:
                return [], [], 0, simple_ata, order

            X_cross.append((start_link + end_link[16:] + [cross_time]))
            # time += cross_time

        return np.array(X_trains, np.float32), np.array(X_cross, np.float32), ata, simple_ata, order

    def gethead(self, line):

        strs = line.split(";;")
        # head
        head_s = strs[0].split(" ")
        order, ata, distance, simple_ata, driver_id, slice_id = head_s[0], float(head_s[1]), float(head_s[2]), float(
            head_s[3]), int(head_s[4]), int(head_s[5])

        return ata, distance, simple_ata, driver_id, slice_id

    def process(self, file_path):

        order_list = []
        time_list = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            np.random.shuffle(lines)

            for line in lines:
                order, time, simple_ata, ata = self.processStr(line.strip())
                print(order, " ", time, " ", simple_ata, " ", ata)
                # order_list.append(order)
                # time_list.append(time)

        # with open("E:/result.csv","w") as file:
        #
        #    for order,time in zip(order_list,time_list):
        #        file.write(str(order)+" "+str(time)+"\n")

#         x = np.array([order_list, time_list])
#         x = np.transpose(x, (1,0))

#         name = ["id", "result"]
#         test=pd.DataFrame(columns=name,data=x)
#         test.to_csv('E:/result.csv',index = False,encoding='utf-8')
#         order_list.clear()
#         time_list.clear()
#         x.clear()
