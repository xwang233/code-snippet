import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)

dtype = torch.float
device = 'cuda'
n = 1
c = 3
h = 256
w = 256
nc = 91

model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=nc)

model = model.to(device=device)
scalar = torch.cuda.amp.GradScaler()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss().to(device=device)
data = torchvision.datasets.FakeData(
    size=10,
    num_classes=nc, 
    transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=n)

for d, t in loader:
    d = d.to(device=device)
    t = t.to(device=device)
    optimizer.zero_grad()
    model.train()

    tds = []
    for _ in range(n):
        td = {}
        td_box, _ = torch.rand(n, 4, dtype=torch.float, device=device).sort(dim=1)
        td['boxes'] = td_box
        # td['boxes'] = td_box.half()
        td['labels'] = torch.randint(0, nc-1, (n, ), dtype=torch.int64, device=device)
        td['masks'] = torch.randint(0, 1, (n, h, w), dtype=torch.uint8, device=device)
        tds.append(td)

    with torch.cuda.amp.autocast():
        loss_dict = model(d, tds)
        loss = sum(s for s in loss_dict.values())
    scalar.scale(loss).backward()
    scalar.step(optimizer)
    scalar.update()

    # loss_dict = model(d, tds)
    # loss = sum(s for s in loss_dict.values())
    # loss.backward()
    # optimizer.step()

torch.cuda.synchronize()