'''模型训练'''
import os
import json
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore.context as context
from model import resnet34


context.set_context(device_target="GPU")

data_transform = {
    "train": ds.transforms.Compose([ds.vision.RandomResizedCrop(224),
                                    ds.vision.RandomHorizontalFlip(),
                                    ds.vision.ToTensor(),
                                    ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": ds.transforms.Compose([ds.vision.Resize(256),
                                  ds.vision.CenterCrop(224),
                                  ds.vision.ToTensor(),
                                  ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


data_root = os.path.abspath(os.path.join(
    os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path

train_dataset = ds.ImageFolderDataset(dataset_dir=image_path+"train",
                                      transform=data_transform,
                                      class_indexing={'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflower': 3, 'tulips': 4})
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = {'daisy': 0, 'dandelion': 1,
               'roses': 2, 'sunflower': 3, 'tulips': 4}
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = ds.GeneratorDataset(train_dataset,
                                   shuffle=True,
                                   num_parallel_workers=0)
train_loader = train_loader.batch(batch_size=batch_size)

validate_dataset = ds.ImageFolderDataset(dataset_dir=image_path + "val",
                                         transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = ds.GeneratorDataset(validate_dataset,
                                      shuffle=False,
                                      num_parallel_workers=0)
validate_loader = validate_loader.batch(batch_size=batch_size)


net = resnet34()
# load pretrain weights
model_weight_path = "./resnet34-pre.ckpt"
param_not_load, _ = mindspore.load_param_into_net(
    net, mindspore.load_checkpoint(model_weight_path))
print(param_not_load)
# missing_keys, unexpected_keys = net.load_state_dict(
# torch.load(model_weight_path), strict=False)
# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure
inchannel = net.fc.in_channels
net.fc = nn.Dense(inchannel, 5)

loss_function = nn.CrossEntropyLoss()
optimizer = nn.Adam(net.trainable_params(), lr=0.0001)

best_acc = 0.0
save_path = './resNet34.ckpt'


def forward_fn(data, label):
    '''前向传播'''
    # Define forward function
    logits = net(data)
    loss = loss_function(logits, label)
    return loss, logits


# Get gradient function
grad_fn = mindspore.value_and_grad(
    forward_fn, None, optimizer.parameters, has_aux=True)


def train_step(data, label):
    '''训练一次'''
    # Define function of one-step training
    (loss, logits), grads = grad_fn(data, label)
    optimizer(grads)
    return loss, logits


for epoch in range(3):
    # train
    net.set_train(True)
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        loss, logits = train_step(images, labels)

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(
            f"train loss: {int(rate*100):^3.0f}%[{a}->{b}]{loss:.4f}")
    print()

    # validate
    net.set_train(False)
    acc = 0.0  # accumulate accurate number / epoch
    for val_data in validate_loader:
        val_images, val_labels = val_data
        # eval model only have last output layer
        outputs = net(val_images)
        # loss = loss_function(outputs, test_labels)
        predict_y = ops.max(outputs, axis=1)[1]
        acc += (predict_y == val_labels).sum().item()
    val_accurate = acc / val_num
    if val_accurate > best_acc:
        best_acc = val_accurate
        mindspore.save_checkpoint(net, save_path)
    print(f'{epoch + 1} train_loss: {(running_loss / step):.3f}  test_accuracy: {(val_accurate):.3f}')

print('Finished Training')
