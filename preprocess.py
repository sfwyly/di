

"""

    预处理： 对每个属性进行归一化处理

"""
from dataloader import *

root = "E:/competion/didi/giscup_2021/train/"
link_path = "E:/competion/didi/giscup_2021/nextlinks.txt"
weather_path = "E:/competion/didi/giscup_2021/weather.csv"
save_path = "E:/competion/didi/"


def preprocess():

    def comp(x_max, x_min, x_list):
        x_max = max(x_max, np.max(x_list))
        x_min = min(x_min, np.min(x_list))

        return x_max, x_min

    dataloader = DataLoader()
    dataloader.init(link_path, weather_path)

    ata_list, distance_list, simple_ata_list, driver_id_list, slice_id_list = [], [], [], [], []

    ata_max = float("-inf")
    ata_min = float("inf")

    distance_max = float("-inf")
    distance_min = float("inf")

    simple_ata_max = float("-inf")
    simple_ata_min = float("inf")

    driver_id_max = float("-inf")
    driver_id_min = float("inf")

    slice_id_max = float("-inf")
    slice_id_min = float("inf")

    for date in range(1, 32):
        d = str(date)
        if len(d) == 1:
            d = "0" + d
        d = "202008" + d
        print("训练 ", d, " 日数据")

        file_path = root + d + ".txt"

        with open(file_path, "r") as file:
            lines = file.readlines()
            np.random.shuffle(lines)

            for line in lines:
                ata, distance, simple_ata, driver_id, slice_id = dataloader.gethead(line.strip())
                ata_list.append(ata)
                distance_list.append(distance)
                simple_ata_list.append(simple_ata)
                driver_id_list.append(driver_id)
                slice_id_list.append(slice_id)
        if len(ata_list)<=0:
            continue
        ata_max, ata_min = comp(ata_max, ata_min, ata_list)
        distance_max, distance_min = comp(distance_max, distance_min, distance_list)
        simple_ata_max, simple_ata_min = comp(simple_ata_max, simple_ata_min, simple_ata_list)
        driver_id_max, driver_id_min = comp(driver_id_max, driver_id_min, driver_id_list)
        slice_id_max, slice_id_min = comp(slice_id_max, slice_id_min, slice_id_list)

        ata_list.clear()
        distance_list.clear()
        simple_ata_list.clear()
        driver_id_list.clear()
        slice_id_list.clear()

        print(ata_max, " ", ata_min)
        print(distance_max, " ", distance_min)
        print(simple_ata_max, " ", simple_ata_min)
        print(driver_id_max, " ", driver_id_min)
        print(slice_id_max, " ", slice_id_min)

    print(ata_max, " ", ata_min) # 11747 11
    print(distance_max, " ", distance_min)# 144247.274 11.6293
    print(simple_ata_max, " ", simple_ata_min)# 11161.0 2.0
    print(driver_id_max, " ", driver_id_min)# 80886 0
    print(slice_id_max, " ", slice_id_min)# 287 0

if __name__ == '__main__':

    dataloader = DataLoader()
    dataloader.init(link_path, weather_path)

    print(len(dataloader.link_dict))