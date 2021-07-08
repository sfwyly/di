
import os
from model import *
from dataloader import *

root = "E:/competion/didi/giscup_2021/20200901_test.txt"
link_path = "E:/competion/didi/giscup_2021/nextlinks.txt"
weather_path = "E:/competion/didi/giscup_2021/weather.csv"
save_path = "E:/competion/didi/"

def test():
    dataloader = DataLoader()
    dataloader.init(link_path, weather_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model1 = Model1()
    # model1.apply(weights_init)
    model1.to(device)
    model2 = Model2()
    # model2.apply(weights_init)
    model2.to(device)

    # if os.path.exists(save_path + "model.h5"):
    #     model_dict = torch.load(save_path + "model.h5", map_location=device)
    #     model1.load_state_dict(model_dict['model1'])
    #     model2.load_state_dict(model_dict['model2'])

    model1.eval()
    model2.eval()
    weather = [0, 0, 0, 0, 0]
    weather[3] = 1 # rainstorm
    week = [0, 0, 0, 0, 0, 0, 0]
    week[2] = 1 # 周二

    result = []

    with open(root,"r") as file:

        for line in file.readlines():

            X_trains, X_cross, y, simple_ata, order = dataloader.processStr(line.strip(), weather, week)

            # if len(X_trains) <= 0:
            #     result.append([order, simple_ata])
            #     continue
            # X_trains = torch.from_numpy(X_trains).to(device, torch.float32)
            # r = torch.sum(model1(X_trains))
            # if len(X_cross) > 0:
            #     X_cross = torch.from_numpy(X_cross).to(device, torch.float32)
            #     r2 = torch.sum(model2(X_cross))
            #     r += r2
            #result.append([order, r.item()])
            result.append([order, simple_ata])
    name = ["id", "result"]
    result = pd.DataFrame(columns=name, data=result)
    result.to_csv(save_path + 'result.csv', index=False, encoding='utf-8')




if __name__ == "__main__":

    test()
