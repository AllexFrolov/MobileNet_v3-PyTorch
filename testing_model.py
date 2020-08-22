import torch
import config
from MobileNet_v3 import get_model
from torchvision import transforms
from datafunc import Dataset, MyDataLoader


def load_model(path='model.torch'):
    """
    Загружает аргументы и веса модели
    :param path: Путь к файлу
    :return: Модель
    """
    checkpoint = torch.load(path)
    model = get_model(config.MODEL_SIZE)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def predict(model, dataloader):
    """
    Делает предсказания
    :param model: Модель
    :param dataloader: Итератор данных
    """
    model.eval()
    predicts = []
    f_names = []
    with torch.no_grad():
        for X, names in dataloader:
            predict_proba = model(X).squeeze(dim=-1)
            predicts += torch.argmax(predict_proba, dim=-1).data
            f_names += names
    labels = dataloader.idx_to_class(predicts)
    result = zip(f_names, labels)
    return result


if __name__ == '__main__':
    transformer = transforms.Compose([
        transforms.Resize(config.IM_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*config.NORMALIZE)
    ])
    dataset = Dataset(config.DATA_FOLDER, transformer)
    loader = MyDataLoader(dataset, config.BATCH_SIZE)
    model = load_model()
    predict(model, loader)

