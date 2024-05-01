import os
import numpy as np
import pandas as pd

import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

train = pd.read_csv('../input/global-wheat-detection/train.csv')

# train.head()

coord = pd.DataFrame(list(train.bbox.apply(lambda x: x[1: -1].split(",")).values), columns=['x1', 'y1', 'w', 'h'])

df = pd.concat([train, coord], axis=1)

df["x1"] = pd.to_numeric(df["x1"])
df["y1"] = pd.to_numeric(df["y1"])
df["w"] = pd.to_numeric(df["w"])
df["h"] = pd.to_numeric(df["h"])

df['x2'] = df['x1'] + df['w']
df['y2'] = df['y1'] + df['h']

df.drop(['bbox', 'width', 'height', 'w', 'h', 'source'], axis=1, inplace=True)

unique_imgs = df.image_id.unique()


class CustDat(torch.utils.data.Dataset):

    def __init__(self, df, unique_imgs, indices):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_name = self.unique_imgs[self.indices[idx]]
        boxes = self.df[self.df.image_id == image_name].values[:, 1:].astype("float")
        img = Image.open("../input/global-wheat-detection/train/" + image_name + ".jpg").convert('RGB')
        labels = torch.ones((boxes.shape[0]), dtype=torch.int64)
        target = {}
        target["boxes"] = torch.tensor(boxes)
        target["label"] = labels
        return T.ToTensor()(img), target


train_idx = random.sample(range(unique_imgs.shape[0]), int(0.8 * unique_imgs.shape[0]))
valid_idx = list(set(range(unique_imgs.shape[0])) - set(train_idx))
print(len(train_idx), len(valid_idx))


def custom_collate(data):
    return data


train_dl = torch.utils.data.DataLoader(CustDat(df, unique_imgs, train_idx), batch_size=16, shuffle=True,
                                       collate_fn=custom_collate,
                                       pin_memory=True if torch.cuda.is_available() else False)

valid_dl = torch.utils.data.DataLoader(CustDat(df, unique_imgs, valid_idx), batch_size=8, shuffle=True,
                                       collate_fn=custom_collate,
                                       pin_memory=True if torch.cuda.is_available() else False)
print(len(train_dl), len(valid_dl))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.1)
num_epochs = 20


epoch_total_losses = []
val_epoch_total_losses = []

model.to(device)
for epoch in range(num_epochs):

    epoch_total_loss = 0
    val_epoch_total_loss = 0
    for data in train_dl:
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(device))
            targ = {}
            targ['boxes'] = d[1]['boxes'].to(device)
            targ['labels'] = d[1]['label'].to(device)
            targets.append(targ)
        loss_dict = model(imgs, targets)
        loss = loss_dict['loss_classifier']
        epoch_total_loss += loss.cpu().detach().numpy()
        epoch_total_loss = epoch_total_loss / len(train_dl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Train Total Loss: {epoch_total_loss}')

    for val_data in valid_dl:
        val_imgs = []
        val_targets = []
        for d in data:
            val_imgs.append(d[0].to(device))
            val_targ = {}
            val_targ['boxes'] = d[1]['boxes'].to(device)
            val_targ['labels'] = d[1]['label'].to(device)
            val_targets.append(targ)
        with torch.no_grad():
            val_loss_dict = model(val_imgs, val_targets)
            val_loss = val_loss_dict['loss_classifier']
            val_epoch_total_loss += val_loss.cpu().detach().numpy()
            val_epoch_total_loss = val_epoch_total_loss / len(valid_dl)
    print(f'Epoch {epoch + 1}, Validation Total Loss: {val_epoch_total_loss}')

    epoch_total_losses.append(epoch_total_loss)
    val_epoch_total_losses.append(val_epoch_total_loss)

plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), epoch_total_losses, label='Train Total Loss')
plt.plot(range(num_epochs), val_epoch_total_losses, label='Validation Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses')
plt.legend()

plt.savefig('losses_eval.png')

model.eval()
data = iter(valid_dl).__next__()

img = data[0][0]
boxes = data[0][1]["boxes"]
labels = data[0][1]["label"]

output = model([img.to(device)])

print(output)

out_bbox = output[0]["boxes"]
out_scores = output[0]["scores"]

keep = torchvision.ops.nms(out_bbox, out_scores, 0.5)

print(out_bbox.shape, keep.shape)

im = (img.permute(1,2,0).cpu().detach().numpy() * 255).astype('uint8')

print(im)

vsample = Image.fromarray(im)

draw = ImageDraw.Draw(vsample)

for box in boxes:
  draw.rectangle(list(box), fill = None, outline = "red", width = 2)

file_path = './output_image.jpg'
vsample.save(file_path)

