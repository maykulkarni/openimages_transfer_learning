import requests
import pandas as pd
from label_image import predict_image
import os


def save_image(url, path):
    try:
        response = requests.get(url)
        print("Downloading: {}".format(url))
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
                print("Saved to: {}".format(path))
        else:
            print("Non 200 Respose: {}".format(url))
    except Exception as e:
        print("Exception: {}".format(e))


def create_urls_list():
    csv_file_path = "links.csv"
    url_df = pd.read_csv(csv_file_path)
    urls_list = url_df["thumbnail_300k_url"].tolist()
    label = url_df["label_display_name"].tolist()
    subset = url_df["subset"].tolist()
    result = [(x, y, z) for x, y, z in zip(urls_list, label, subset)]
    return result


def get_folder_name(label_name):
    map = {"Lemon": "lemon",
           "Banana": "banana",
           "Dolphin": "dolphin",
           "Sea lion": "sea_lion",
           "Baseball bat": "baseball_bat"}
    return map[label_name]


def download_images():
    urls = create_urls_list()
    counts = {"Lemon": 0, "Banana": 0, "Dolphin": 0, "Sea lion": 0, "Baseball bat": 0}
    for image in urls:
        url, label, subset = image
        path = "images/{subset}/{label}/{count}.jpg".format(subset=subset,
                    label=get_folder_name(label), count=counts[label])
        counts[label] += 1
        save_image(url, path)


def test_accuracy():
    # image = "images/test/banana/0.jpg"
    labels = ["banana", "lemon", "sea_lion", "dolphin", "baseball_bat"]

    # banana: 4 / 64
    # baseball_bat: 2 / 36
    # dolphin: 4 / 120
    # lemon: 2 / 175
    # sea_lion: 1 / 82
    # ------------------
    # 13 /
    total_images = 0
    correct = 0
    for label in labels:
        base_path = "images/test/{}/".format(label)
        for index, test_images in enumerate(os.listdir(base_path)):
            y_pred = predict_image(base_path + test_images)
            if y_pred == label:
                correct += 1
                msg = "Correct Prediction!"
            else:
                msg = "Wrong Prediction!"
            print("{}: Actual: {} Prediction: {} Msg: {}".format(str(index), label, y_pred, msg))
            total_images += 1
            print("Accuracy: {0:.2f}%".format((correct / total_images) * 100))
    # predict_image(image)


if __name__ == "__main__":
    test_accuracy()
    # download_images()
