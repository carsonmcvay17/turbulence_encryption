# imports
import torch
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
from turbencrypt.FNO import FourierNO


# from Training import Trainer2

data_path = "/Users/gilpinlab/turbulence_encryption/data.npz"
random_state = 42
test_size = .2
device = 'cpu'
loaders = FourierNO.makeFNO(data_path, random_state, test_size)
train_loader = loaders[0]
test_loader = loaders[1]


model = FNO(n_modes=(16,16),
           in_channels=1,
           out_channels=1, 
           hidden_channels=32,
           projection_channel_ratio=2)

model = model.to(device)

optimizer = AdamW(model.parameters(),
                 lr=1e-4,
                 weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# create losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}

# create the trainer
trainer = Trainer(model=model, n_epochs=20, 
                 wandb_log=False,
                 device=device,
                 mixed_precision=False,
                 data_processor=None,
                 eval_interval=3,
                 log_output=False,
                 use_distributed=False,
                 verbose=True)

# train model on data
trainer.train(train_loader=train_loader, 
             test_loaders={"default": test_loader},
             optimizer=optimizer,
             scheduler=scheduler,
             regularizer=False,
             training_loss=train_loss,
             eval_losses=eval_losses)