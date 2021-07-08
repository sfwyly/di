
from model import *
from dataloader import *
import os

root = "E:/competion/didi/giscup_2021/train/"
link_path = "E:/competion/didi/giscup_2021/nextlinks.txt"
weather_path = "E:/competion/didi/giscup_2021/weather.csv"
save_path = "E:/competion/didi/"


def train(epochs=100):
    for i in range(epochs):
        print(" 训练 ",i + 1," / ", epochs)
        trainer()


def trainer():

    dataloader = DataLoader()
    dataloader.init(link_path, weather_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model1 = Model1()
    # model1.apply(weights_init)
    model1.to(device)
    model2 = Model2()
    # model2.apply(weights_init)
    model2.to(device)

    if os.path.exists(save_path + "model.pth"):
        model_dict = torch.load(save_path + "model.pth", map_location=device)
        model1.load_state_dict(model_dict['model1'])
        model2.load_state_dict(model_dict['model2'])

    optimizer = torch.optim.Adam([{"params": model1.parameters()}, {"params": model2.parameters()}], lr= 0.0001, betas=(0.5, 0.999))

    model1.train()
    model2.train()

    # 训练 8 月份数据
    for date in range(1, 32):
        d = str(date)
        if len(d) == 1:
            d = "0" + d
        d = "202008" + d
        print("训练 ", d, " 日数据")

        file_path = root + d + ".txt"

        weather = [0, 0, 0, 0, 0]
        weather[dataloader.day_weather[d]] = 1
        week = [0, 0, 0, 0, 0, 0, 0]
        week[dataloader.week_dict[(date % 7)]] = 1
        with open(file_path,"r") as file:
            lines = file.readlines()
            np.random.shuffle(lines)

            for line in lines:

                X_trains, X_cross, y, _, __ = dataloader.processStr(line.strip(), weather, week)

                optimizer.zero_grad()
                loss = 0.
                if len(X_trains) <= 0:
                    continue
                X_trains = torch.from_numpy(X_trains).to(device, torch.float32)
                r = torch.sum(model1(X_trains))
                if len(X_cross) > 0:
                    X_cross = torch.from_numpy(X_cross).to(device, torch.float32)
                    r2 = torch.sum(model2(X_cross))
                    r += r2
                #y = torch.from_numpy(y).to(device, torch.float32)
                y = torch.tensor(y,dtype=torch.float32).to(device, torch.float32)
                loss = l1_loss(r , y)
                loss.backward()
                optimizer.step()
                print(" 训练损失 ： ", loss.item())

        print("保存模型")
        model_dict = {
            'model1' : model1.state_dict(),
            'model2' : model2.state_dict()
        }
        torch.save(model_dict, save_path + "model.pth")

if __name__ =="__main__":

    train()