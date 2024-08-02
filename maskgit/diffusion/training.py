# Imports
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
from jax.tree_util import tree_map
import optax as opt

# For the trainer
from time import time
from tqdm import tqdm, trange
from copy import deepcopy
import jax_dataloader as jdl
import flax.serialization
import os, shutil
import pickle as pkl

# Typing
from typing import Any, Callable, Sequence, Iterable, Dict, NamedTuple
from chex import Array, Scalar, ArrayTree

# from src.ema import ema

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any

def get_checkpoint_name(config):
    return config["experiment_name"] + "_" + config["wandb_run_id"]

def save_checkpoint(state, checkpoint_dir, step):
    """
    state: dictionary {
        params: the current model parameters
        config: the training configuration
        opt_state: the state of the optimizer
    }
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # List all files in the directory
    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)

        # Check if this file is a savepoint (customize this condition as needed)
        if filename.startswith("checkpoint_"):
            # Delete file or directory
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                # Check if we want to keep this checkpoint
                stripped_name = filename.replace("checkpoint_", "").replace(".pkl", "")
                try:
                    number = int(stripped_name)
                    if number > 0 and number % 1000 == 0:
                        pass
                    else:
                        os.remove(file_path)
                except:
                    pass
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pkl')
    with open(checkpoint_path, "wb") as file:
        pkl.dump(state, file)
        
def load_checkpoint(checkpoint_dir, step):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pkl')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as file:
            state = pkl.load(file)
        return state
    else:
        return None

class Trainer:
    """
    model: doesn't have to be a pytree node
    loss (key, params, model, data, **config) -> (loss, aux)
        Returns a loss (a single float) and an auxillary output (e.g. posterior)
    init (key, model, data, **config) -> (params, opts)
        Returns the initial parameters and optimizers to go with those parameters
    update (params, grads, opts, model, aux, **config) -> (params, opts)
        Returns updated parameters, optimizers
    """

    def __init__(self, model: Any,
                 config: Dict = None,
                 init: Callable = None,
                 loss: Callable = None,
                 val_loss: Callable = None,
                 update: Callable = None,
                 initial_params: Dict = None):
        # Trainer state
        self.params = initial_params
        self.model = model
        self.past_params = []
        self.time_spent = []

        if config is None:
            config = dict()

        self.config = config

        if init is not None:
            self.init = init
        if loss is not None:
            self.loss = loss

        self.val_loss = val_loss or self.loss
        if update is not None:
            self.update = update

    def train_step(self, key, params, batch, opt_states, itr):
        model = self.model

        results = jax.value_and_grad(
            lambda _params: self.loss(params=_params,
                                      key=key,
                                      model=model,
                                      batch=batch,
                                      itr=itr,
                                      **self.config),
            has_aux=True)(params)

        (loss, aux), grads = results
        params, opts = self.update(params, grads, self.opts, opt_states, model, aux, self.config)
        return params, opts, (loss, aux), grads

    def val_step(self, key, params, batch):
        return self.val_loss(params=params,
                             key=key,
                             model=self.model,
                             batch=batch, **self.config)

    def val_epoch(self, key, params, val_loader):

        epoch_summary = []

        for (batch_data,) in iter(val_loader):

            loss_out = self.val_step_jitted(key, params, batch_data)
            epoch_summary.append(tree_map(lambda x: jnp.mean(x), loss_out))

        epoch_summary = tree_map(lambda *arrays: jnp.stack(arrays), *epoch_summary)
        return epoch_summary

    """
    Callback: a function that takes training iterations and relevant parameter
        And logs to WandB
    """
    
    def save_checkpoint(self, epoch):
        """
        state: dictionary {
            params: the current model parameters
            config: the training configuration
            opt_state: the state of the optimizer
        }
        """
        name = get_checkpoint_name(self.config)
        state = {"params": self.params, 
                 "opt_state": self.opt_states, 
                 "config": self.config }
        checkpoint_dir = "../saves/"+ name + "/checkpoints"
        save_checkpoint(state, checkpoint_dir, epoch)

    def train(self, data_dict, max_epochs,
              callback=None, val_callback=None,
              summary=None, key=None,
              early_stop_start=10000,
              max_lose_streak=1000,
              start_epoch=0):

        if key is None:
            key = jr.PRNGKey(0)

        model = self.model
        train_data = data_dict["train_data"]
        batch_size = self.config.get("batch_size") or train_data.shape[0]
        
        use_validation = self.config.get("use_validation")
        
        # Create dataloaders with jax_dataloader
        train_loader = jdl.DataLoader(jdl.ArrayDataset(train_data), backend="jax",
                                      batch_size=batch_size,
                                      shuffle=self.config.get("shuffle_dataset"))
        
        if use_validation:
            val_loader = jdl.DataLoader(jdl.ArrayDataset(data_dict["val_data"]), backend="jax",
                                      batch_size=batch_size, shuffle=False)
        val_interval = self.config.get("validation_interval") or 1
        max_iters = max_epochs * len(train_loader)

        init_key, key = jr.split(key, 2)

        # Initialize optimizer
        self.params, self.opts, self.opt_states = self.init(init_key, model,
                                                            train_data[:batch_size],
                                                            self.params,
                                                            self.config)
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []
        self.past_params = []

        train_step = jit(self.train_step)
        self.val_step_jitted = jit(self.val_step)

        itr = 0
        best_loss = None
        best_epoch = 0
        val_loss = None
        curr_loss = jnp.inf

        show_progress = not self.config.get("slient_training")
        if (show_progress): pbar = tqdm(total=max_iters, desc="[jit compling...]")

        checkpoint_freq = self.config.get("checkpoint_frequency") or 1000000
            
        for epoch in range(start_epoch+1, max_epochs):
            train_key, val_key, key = jr.split(key, 3)

            epoch_summary = []
            for (train_batch,) in iter(train_loader):

                batch_key, train_key = jr.split(train_key)
                t = time()
                # Training step
                # ----------------------------------------
                step_results = train_step(batch_key, self.params,
                                          train_batch, self.opt_states, itr)
                self.params, self.opt_states, loss_out, grads = \
                    jax.tree_map(lambda x: x.block_until_ready(), step_results)
                # ----------------------------------------
                dt = time() - t
                self.time_spent.append(dt)

                loss, _ = loss_out
                self.train_losses.append(loss)
                epoch_summary.append(tree_map(lambda x: jnp.mean(x), loss_out))
                # Update the progressbar every iteration
                itr += 1
                if (show_progress):
                    pbar.set_description(f"Iteration {itr + 1}, epoch {epoch + 1}/{max_epochs}, prev epoch loss: {curr_loss:.3f}")
                    pbar.update(1)

            # Train epoch callback
            epoch_summary = tree_map(lambda *arrays: jnp.stack(arrays), *epoch_summary)
            epoch_losses, _ = epoch_summary

            if (callback): callback(self, epoch_summary, data_dict, grads)
            
            # Validation/test epoch and callback
            if use_validation and epoch % val_interval == 0:
                val_loss_out = self.val_epoch(val_key, self.params, val_loader)
                if (val_callback): val_callback(self, val_loss_out, data_dict)
                val_loss, _ = val_loss_out
                self.val_losses.append(val_loss.mean())
            else:
                val_loss = None

            # Checkpointing
            if epoch % checkpoint_freq == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping based on loss improvement
            if not use_validation or val_loss is None:
                curr_loss = epoch_losses.mean()
            else:
                curr_loss = val_loss.mean()

            if epoch >= early_stop_start:
                if best_loss is None or curr_loss < best_loss:
                    best_epoch = epoch
                    best_loss = curr_loss
                if curr_loss > best_loss and epoch - best_epoch > max_lose_streak:
                    if show_progress:
                        pbar.set_description(pbar.desc +" Early stopping!")
                        pbar.close()
                    else:
                        print("Early stopping!")
                    break
                    
        self.save_checkpoint(epoch)
        
        if show_progress:
            pbar.close()

        if summary:
            summary(self, data_dict)

def generic_model_init(key: PRNGKey,
                       model: Any,
                       init_data: Array,
                       init_params: Dict = None,
                       config: Dict = None):
    """
    Generic model initialization with an Adam optimizer from optax.
    """
    if init_params is None:
        init_params = model.init(key, init_data[0])
    model_opt = opt.adam(learning_rate=config["learning_rate"])
    opt_state = model_opt.init(init_params)
    return init_params, model_opt, opt_state

def linear_warmup_schedule(warmup_steps, initial_lr, peak_lr):
    """
    Returns a schedule that linearly ramps up the learning rate to peak_lr over warmup_steps,
    then stays at peak_lr.
    """
    def schedule(step):
        return jnp.where(step < warmup_steps,
                         initial_lr + step / warmup_steps * (peak_lr - initial_lr),
                         peak_lr)
    return schedule

def model_init(
        key, model,
        init_data, init_params = None,
        config = None):
    """
    Model initialization that adds a copy of the parameters to config.
    """
    if init_params is None:
        params_key, dropout_key = jr.split(key)
        init_params = model.init({"params": params_key, 
                                  "dropout": dropout_key}, init_data[0], 0)
    
    lr_schedule = linear_warmup_schedule(5000, 0, config["learning_rate"])
    
    optimizer = opt.chain(
        opt.clip_by_global_norm(1.0),  # Clip gradients to have a global norm of at most `clip_norm`
        opt.adam(lr_schedule),
    )
    # model_opt = ema(optimizer, config["ema_decay"])
    opt_state = model_opt.init(init_params)
    return init_params, model_opt, opt_state

def generic_params_update(model_params: Dict,
                          grad: ArrayTree,
                          model_opt: Any,
                          opt_state: Any,
                          model: Any,
                          aux: Any,
                          config: Dict):
    """
    Generic gradient update with optax optimizers.
    """
    updates, opt_state = model_opt.update(grad, opt_state, params=model_params)
    model_params = opt.apply_updates(model_params, updates)
    return model_params, opt_state

# def time_condtioned_model_init_ema(key: PRNGKey,
#                        model: Any,
#                        init_data: Array,
#                        init_params: Dict = None,
#                        config: Dict = None):
#     """
#     Model initialization that adds a copy of the parameters to config.
#     """
#     if init_params is None:
#         init_params = model.init(key, init_data[0], 0)
#     model_opt = ema(opt.adam(learning_rate=config["learning_rate"]), config["ema_decay"])
#     opt_state = model_opt.init(init_params)
#     return init_params, model_opt, opt_state

# def time_condtional_model_init_dropout_ema(
#         key: PRNGKey,
#         model: Any,
#         init_data: Array,
#         init_params: Dict = None,
#         config: Dict = None):
#     """
#     Model initialization that adds a copy of the parameters to config.
#     """
#     if init_params is None:
#         params_key, dropout_key = jr.split(key)
#         init_params = model.init({"params": params_key, 
#                                   "dropout": dropout_key}, init_data[0], 0)
#     model_opt = ema(opt.adam(learning_rate=config["learning_rate"]), config["ema_decay"])
#     opt_state = model_opt.init(init_params)
#     return init_params, model_opt, opt_state

def empty_callback(trainer, loss_out, data_dict, grads):
	pass