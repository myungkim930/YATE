## Load data
from data import Load_data

main_data = Load_data(data_name="yago310")

## Define model and parameters
from models import YATE_Encode
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from graphlet_construction import make_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = YATE_Encode(
    input_dim=300, emb_dim=300, output_dim=100, num_heads=1, num_layers=2
)
model = model.to(device)

n_batch = 100
num_hops = 1
n_epoch = 500
n_pos = 5
n_neg = 2

weight = torch.tensor(
    [
        (n_pos + 1) * n_neg / ((n_pos + 1) + (n_pos + 1) * n_neg),
        (n_pos + 1) / ((n_pos + 1) + (n_pos + 1) * n_neg),
    ],
    device=device,
)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(weight=weight)
scheduler2 = optim.lr_scheduler.LinearLR(optimizer)

T_0 = int(main_data.edgelist_total.size()[1] / n_batch * n_epoch)
scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)

ent_list = main_data.edge_index[0, :]
idx_data = DataLoader(ent_list, batch_size=n_batch, shuffle=True)

# ## train
# temp = next(iter(idx_data))
# idx_cen = temp.clone()

model.train()

for _ in range(n_epoch):

    with tqdm(n_epoch, desc="(T)") as pbar:

        for idx in idx_data:

            data = make_batch(
                idx_cen=idx_cen,
                num_hops=num_hops,
                main_data=main_data,
                n_pos=n_pos,
                per_pos=0.8,
                n_neg=n_neg,
                per_neg=0.8,
            )

            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            target = data.y.type(torch.int64)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            scheduler1.step()

        pbar.set_postfix({"loss": loss})
        pbar.update()

        scheduler2.step()

    for _ in range(n_epoch):

        optimizer.zero_grad()

        # target = torch.reshape(data.y, (data.y.size()[0], 1))

        pbar.set_postfix({"loss": loss})
        pbar.update()


# device = "cpu"
# pos_weight = torch.tensor([n_pos / (n_pos + n_pos * n_neg)], device=device)
# T_0 = int(YATE_data.edgelist_total.size()[1] / n_batch * n_epoch)
# scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)
# scheduler2 = optim.lr_scheduler.LinearLR(optimizer)

# scheduler1.step()
# scheduler2.step()

# for name, param in model.named_parameters():
#     print(name, param.grad.norm())

# for p,n in zip(rnn.parameters(),rnn._all_weights[0]):
#     if n[:6] == 'weight':
#         print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))
