# imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
from turbencrypt.FNO import FourierNO
from neuralop.data.transforms.data_processors import IncrementalDataProcessor
from neuralop.training.incremental import IncrementalFNOTrainer



# from Training import Trainer2

data_path = "/Users/gilpinlab/turbulence_encryption/data/mnist_re700_g64.npz"
random_state = 42
test_size = 0.2
device = 'cpu'
train_loader, test_loader, data_processor = FourierNO.makeFNO(data_path, random_state, 16, test_size)




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
num_iters = 80
optimizer = AdamW(model.parameters(),
                 lr=8e-3, weight_decay=1e-4) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
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
    n_epochs=num_iters,
    data_processor=data_transform,
    device=device,
    verbose=True,
    incremental_loss_gap=False,
    incremental_grad=True,
    incremental_grad_eps=0.9999,
    incremental_loss_eps = 0.001,
    incremental_buffer=5,
    incremental_max_iter=num_iters,
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


# evaluation
num_samples = 4
model.eval()
input_list = []
target_list = []
output_list = []
with torch.no_grad():
    for batch in test_loader:
        
        processed_batch = data_processor.preprocess(batch)
        input_data = processed_batch['x']
        target_data = processed_batch['y']

        input_data, target_data = input_data.to(device), target_data.to(device)


        # make predictions
        output = model(input_data)

        h1losses = h1loss(output, target_data).item()
        lplosses = l2loss(output, target_data).item()

        # store data
        input_list.append(input_data.cpu().numpy())  # Convert to NumPy arrays on CPU
        target_list.append(target_data.cpu().numpy())
        output_list.append(output.cpu().numpy())
        # visualize
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 10))



        for axes_row, idx in zip(axes, np.random.choice(len(input_data), num_samples, replace=False)):
            for ax, data, label in zip(axes_row, [input_data[idx], target_data[idx], output[idx]], ['Input', 'Target', 'Prediction']):
                im = ax.imshow(data[0].cpu().numpy(), cmap='seismic')
                ax.set_title(label)
                ax.axis('off')
                fig.colorbar(im, shrink=0.5)

        fig.suptitle(f"batch losses (h1) : {h1losses:.4f} (l2) : {lplosses:.4f}")
        plt.tight_layout()
        plt.show()  

    input_data_array = np.concatenate(input_list, axis = 0)
    target_data_array = np.concatenate(target_list, axis = 0)
    output_data_array = np.concatenate(output_list, axis = 0)

    # save as .npz
    save_path = f"data/mnist_re700_g64_eval.npz"
    np.savez(save_path, inputs = input_data_array, targets = target_data_array, outputs = output_data_array)
    print("Data saved")

        