import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization


from mbv2 import mobilenet_v2
from dataset import get_dataset
from val import val


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    # liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs), flush=True)
        print("-" * 10, flush=True)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            print(f"\n--- phase : {phase} ---\n", flush=True)
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if i % 10 == 0:
                    print(
                        "\rIteration: {}/{}, Loss: {}, LR: {} ".format(
                            i + 1, len(dataloaders[phase]), loss.item() * inputs.size(0),
                            optimizer.param_groups[0]['lr']
                        ), flush=True
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("Train Loss: {:.4f} Acc: {:.4f}".format(avg_loss, t_acc), flush=True)
        print("Val Loss: {:.4f} Acc: {:.4f}".format(val_loss, val_acc), flush=True)
        print("Best Val Accuracy: {}".format(best_acc), flush=True)
        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        ), flush=True
    )
    print("Best val Acc: {:4f}".format(best_acc), flush=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = mobilenet_v2(num_classes=200)
# model = mobilenet_v2(num_classes=200, norm_layer=FrozenBatchNorm2d)
# weight_imagenet = torch.load("mobilenet_v2-b0353104.pth")
# weight_imagenet.pop("classifier.1.weight")
# weight_imagenet.pop("classifier.1.bias")
# model.load_state_dict(weight_imagenet, strict=False)

checkpoint = torch.load("models/mbv2_fp16.pth")
ckpt = {}
for k, v in checkpoint.items():
    if k.startswith('module.'):
        ckpt[k[7:]] = v
    else:
        ckpt[k] = v
model.load_state_dict(ckpt, strict=True)

model.train()


# prepare
#########################################
extra_qconfig_dict = {
    'w_observer': 'MSEObserver',
    'a_observer': 'EMAMSEObserver',
    'w_fakequantize': 'FixedFakeQuantize',
    'a_fakequantize': 'FixedFakeQuantize',
}
# prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
prepare_custom_config_dict = {
    'extra_qconfig_dict': extra_qconfig_dict
}
model = prepare_by_platform(model, BackendType.Tensorrt, prepare_custom_config_dict)
print(model, flush=True)
#########################################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Multi GPU
model = torch.nn.DataParallel(model, device_ids=[0, 1])

# Loss Function
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

train_dataset, val_dataset, _ = get_dataset()

train_loaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=8
)
val_loaders = torch.utils.data.DataLoader(
    val_dataset, batch_size=128, shuffle=True, num_workers=8
)


dataloaders = {}
dataloaders["train"] = train_loaders
dataloaders["val"] = val_loaders

dataset_sizes = {}
dataset_sizes["train"] = len(train_dataset)
dataset_sizes["val"] = len(val_dataset)


print("*" * 100)

# ptq
#########################################
model.eval()
enable_calibration(model)

for i, (inputs, labels) in enumerate(dataloaders["train"]):
    inputs = inputs.to(device)
    with torch.no_grad():
        model(inputs)
    print("\rPTQ: Iteration: {}/{} ".format(i + 1, len(dataloaders["train"])), flush=True)
    if i == 4:
        break

enable_quantization(model)
model.train()
#########################################


# val ptq
#########################################
model.eval()

val(model, val_dataset, val_loaders)

model.train()
#########################################


model = train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=3,
)

model.eval()
torch.save(model.state_dict(), "models/mbv2_tiny_imagenet_fixed_fbn_v2.pth")

convert_deploy(model.module, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='mbv2_tiny_imagenet_fixed_mse_fbn_v2')


