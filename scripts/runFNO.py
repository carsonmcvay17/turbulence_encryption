# imports
import torch
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
from turbencrypt.FNO import FourierNO
from neuralop.data.transforms.data_processors import IncrementalDataProcessor
from neuralop.training.incremental import IncrementalFNOTrainer



# from Training import Trainer2

data_path = "/Users/gilpinlab/turbulence_encryption/data/forreal2.npz"
random_state = 42
test_size = 0.2
device = 'cpu'
train_loader, test_loader, data_processor = FourierNO.makeFNO(data_path, random_state, test_size)


model = FNO(
    max_n_modes=(64,64),
    n_modes=(2,2),
           in_channels=1,
           out_channels=1, # looking at num of channels
           hidden_channels=32,
           )

model = model.to(device)


data_transform = IncrementalDataProcessor(
    in_normalizer=data_processor.in_normalizer,
    out_normalizer=data_processor.out_normalizer,
    device=device,
    subsampling_rates=[4, 2, 1],
    dataset_resolution=64,
    dataset_indices=[2, 3],
    epoch_gap=200,
    verbose=True,
)

data_transform = data_transform.to(device)


optimizer = AdamW(model.parameters(),
                 lr=8e-3, weight_decay=1e-3) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.0)

# create losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = l2loss
eval_losses={'l2': l2loss, 'h1': h1loss}


# create the trainer
# trainer = Trainer(model=model, n_epochs=20, 
#                  wandb_log=False,
#                  device=device,
#                  data_processor=data_processor,
#                  eval_interval=1,
#                  log_output=False,
#                  use_distributed=False,
#                  verbose=True)
trainer = IncrementalFNOTrainer(
    model=model,
    n_epochs=18,
    data_processor=data_transform,
    device=device,
    verbose=True,
    incremental_loss_gap=False,
    incremental_grad=True,
    incremental_grad_eps=0.9999,
    incremental_loss_eps = 0.001,
    incremental_buffer=5,
    incremental_max_iter=18,
    incremental_grad_max_iter=2,
)

# train model on data
trainer.train(train_loader=train_loader, 
             test_loaders={"problem": test_loader},
             optimizer=optimizer,
             scheduler=scheduler,
             regularizer=False,
             training_loss=train_loss,
             eval_losses=eval_losses)