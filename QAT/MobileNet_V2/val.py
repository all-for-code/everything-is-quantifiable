import torch


def val(model, val_dataset, dataloaders):
    running_corrects = 0
    for i, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    print(f"Accuracy : {running_corrects / len(val_dataset) * 100}%")
