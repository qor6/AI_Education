import pandas as pd
import numpy as np
import time

class DataLoader:
    def __init__(self):
        return

    @staticmethod
    def load_height_weight(file_name):
        df = pd.read_csv(file_name, sep=',', header=0)
        data_arr = df.to_numpy()
        heights = data_arr[:, 1]
        weights = data_arr[:, 2]
        # convert inch to cm, lb to kg
        heights *= 2.54
        weights *= 0.453592
        return heights, weights

    @staticmethod
    def load_wine_quality(file_name):
        df = pd.read_csv(file_name, sep=';', header=0)
        data_arr = df.to_numpy()
        features = data_arr[:, 0:-1]
        quality = data_arr[:, -1]
        return features, quality

    @staticmethod
    def load_overfit_example():
        x = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        y = [7.9, 7.7, 8.5, 9.8, 10.1, 9.0, 10.5, 10.4]
        x = np.array(x)
        y = np.array(y)
        return x, y

    @staticmethod
    def convert_forest_fire_data(record):
        num_elements = len(record)
        new_list = []
        for i in range(0, num_elements):
            if i == 0:
                one_hot_vec = []
                for j in range(1, 13):
                    if record[i] == j:
                        one_hot_vec.append(1)
                    else:
                        one_hot_vec.append(0)
                for k in one_hot_vec:
                    new_list.append(k)
            elif i == 1:
                one_hot_vec = []
                for j in range(1, 8):
                    if record[i] == j:
                        one_hot_vec.append(1)
                    else:
                        one_hot_vec.append(0)
                for k in one_hot_vec:
                    new_list.append(k)
            else:
                new_list.append(record[i])

        ret_feature = np.array(new_list)
        num_features = len(ret_feature)
        ret_feature = ret_feature.reshape(-1, num_features)
        return ret_feature, num_features

    def convert_forest_fire_data2(record):
        num_elements = len(record)
        new_list = []
        for i in range(0, num_elements):
            if i == 0:
                one_hot_vec = []
                for j in range(1, 13):
                    if record[i] == j:
                        one_hot_vec.append(1)
                    else:
                        one_hot_vec.append(0)
                for k in one_hot_vec:
                    new_list.append(k)
            else:
                new_list.append(record[i])

        ret_feature = np.array(new_list)
        num_features = len(ret_feature)
        ret_feature = ret_feature.reshape(-1, num_features)
        return ret_feature, num_features

    @staticmethod
    def load_forest_fire(file_name):
        df = pd.read_csv(file_name, sep=',', header=0)
        data_arr = df.to_numpy()
        np.random.seed(seed=0)
        np.random.shuffle(data_arr)

        features = data_arr[:, 2:-1]
        num_features = features.shape[1]
        area = data_arr[:, -1]

        features2 = []
        num_features = 0
        for record in features:
            converted_array, num_features = DataLoader.convert_forest_fire_data(record)
            features2.append(converted_array)
        features2 = np.array(features2)
        features2 = features2.reshape(-1, num_features)
        return features2, area

    @staticmethod
    def load_forest_fire2(file_name):
        df = pd.read_csv(file_name, sep=',', header=0)
        data_arr = df.to_numpy()
        seed = int(time.time())
        np.random.seed(seed=seed)
        np.random.shuffle(data_arr)

        features = data_arr[:, 0:-1]
        num_features = features.shape[1]
        area = data_arr[:, -1]
        fire = []
        for fire_area in area:
            if fire_area > 0.0:
                fire.append(np.log(1+fire_area))
            else:
                fire.append(0)
        fire = np.array(fire)


        features2 = []
        num_features = 0
        for record in features:
            converted_array, num_features = DataLoader.convert_forest_fire_data2(record)
            features2.append(converted_array)
        features2 = np.array(features2)
        features2 = features2.reshape(-1, num_features)
        return features2, fire

    @staticmethod
    def load_forest_fire2_classify(file_name):
        df = pd.read_csv(file_name, sep=',', header=0)
        data_arr = df.to_numpy()
        seed = int(time.time())
        np.random.seed(seed=seed)
        np.random.shuffle(data_arr)

        features = data_arr[:, 0:-1]
        num_features = features.shape[1]
        area = data_arr[:, -1]
        fire = []
        for fire_area in area:
            if fire_area > 0.0:
                fire.append(1)
            else:
                fire.append(0)
        fire = np.array(fire)

        features2 = []
        num_features = 0
        for record in features:
            converted_array, num_features = DataLoader.convert_forest_fire_data2(record)
            features2.append(converted_array)
        features2 = np.array(features2)
        features2 = features2.reshape(-1, num_features)
        return features2, fire

    @staticmethod
    def load_forest_fire3(file_name):
        df = pd.read_csv(file_name, sep=',', header=0)
        data_arr = df.to_numpy()
        seed = int(time.time())
        np.random.seed(seed=seed)
        np.random.shuffle(data_arr)

        features = data_arr[:, 1:-1]
        num_features = features.shape[1]
        area = data_arr[:, -1]
        fire = []
        for fire_area in area:
            if fire_area > 0.0:
                fire.append(np.log(1+fire_area))
            else:
                fire.append(0)
        fire = np.array(fire)

        return features, fire

    @staticmethod
    def load_forest_fire3_classify(file_name):
        df = pd.read_csv(file_name, sep=',', header=0)
        data_arr = df.to_numpy()
        seed = int(time.time())
        np.random.seed(seed=seed)
        np.random.shuffle(data_arr)

        features = data_arr[:, 1:-1]
        num_features = features.shape[1]
        area = data_arr[:, -1]
        fire = []
        for fire_area in area:
            if fire_area > 0.0:
                fire.append(1)
            else:
                fire.append(0)
        fire = np.array(fire)

        return features, fire

    @staticmethod
    def load_cancer_reg(file_name):
        df = pd.read_csv(file_name, sep=',', header=0)
        data_arr = df.to_numpy()
        np.random.shuffle(data_arr)

        features = data_arr[:, 0:-1]
        death_rate = data_arr[:, -1]

        return features, death_rate
