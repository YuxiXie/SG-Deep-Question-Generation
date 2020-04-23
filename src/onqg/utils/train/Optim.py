from math import sqrt
import functools

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def build_torch_optimizer(model, opt):
    """Builds the PyTorch optimizer.
    Input:
        model: The model to optimize.
        opt: The dictionary of options.
    Output:
        A ``torch.optim.Optimizer`` instance.
    """
    params = list(filter(lambda p: p.requires_grad, model.parameters()))   # [p for p in model.parameters() if p.requires_grad]
    betas = [0.9, 0.999]  # adam_beta1 & adam_beta2
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, lr=opt.learning_rate)
    elif opt.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=opt.learning_rate,
                                  initial_accumulator_value=opt.adagrad_accumulator_init)
    elif opt.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=opt.learning_rate)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params, lr=opt.learning_rate, betas=betas, eps=1e-9)
    elif opt.optim == 'fusedadam':
        import apex
        optimizer = apex.optimizers.FusedAdam(params, lr=opt.learning_rate, betas=betas)
    else:
        raise ValueError('Invalid optimizer type: ' + opt.optim)
    
    return {'optim':optimizer, 'para':params}
    
      
def make_learning_rate_decay_fn(opt):
    """Returns the learning decay function from options."""
    if opt.decay_method == 'noam':
        return functools.partial(noam_decay, warmup_steps=opt.n_warmup_steps, model_size=opt.d_model)
    elif opt.decay_method == 'noamwd':
        return functools.partial(noamwd_decay, warmup_steps=opt.n_warmup_steps, model_size=opt.d_model,
                                 rate=opt.learning_rate_decay, decay_steps=opt.decay_steps, start_step=opt.start_decay_steps)
    elif opt.decay_method == 'rsqrt':
        return functools.partial(rsqrt_decay, warmup_steps=opt.n_warmup_steps)
    elif opt.start_decay_steps is not None:
        return functools.partial(exponential_decay, rate=opt.learning_rate_decay, decay_steps=opt.decay_steps, start_step=opt.start_decay_steps)


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in https://arxiv.org/pdf/1706.03762.pdf. """
    return (model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps**(-1.5)))

def noamwd_decay(step, warmup_steps, model_size, rate, decay_steps, start_step=0):
    """Learning rate schedule optimized for huge batches"""
    return (model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps**(-1.5)) *
            rate ** (max(step - start_step + decay_steps, 0) // decay_steps))

def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate` every :obj:`decay_steps` steps. """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)

def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / sqrt(max(step, warmup_steps))


class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin wrapper for `optim`, 
    but also useful for implementing rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such as grad manipulations.
    """
    def __init__(self, optimizer_dict, learning_rate_decay_method, learning_rate, learning_rate_decay=0.5, 
                 lr_decay_step=1, start_decay_steps=5000, learning_rate_decay_fn=None, max_grad_norm=None, 
                 max_weight_value=None, decay_bad_cnt=None):
        self._optimizer = optimizer_dict['optim']
        self._params = optimizer_dict['para']
        self._learning_rate_decay_method = learning_rate_decay_method
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._learning_rate_decay_fn = learning_rate_decay_fn
        self._max_grad_norm = max_grad_norm or 0
        self._max_weight_value = max_weight_value
        self._training_step = 0
        self._decay_step = lr_decay_step
        self._bad_cnt = 0
        self._decay_bad_cnt = decay_bad_cnt
        self._start_decay_steps = start_decay_steps
    
    @classmethod
    def from_opt(cls, model, opt, checkpoint=None):
        """Builds the optimizer from options.
        Input:
            cls: The ``Optimizer`` class to instantiate.
            model: The model to optimize.
            opt: The dict of user options.
            checkpoint: An optional checkpoint to load states from.  
        Output:
            An ``Optimizer`` instance.
        """
        optim_opt = opt
        # optim_state_dict = None

        optimizer = cls(build_torch_optimizer(model, optim_opt),
                        optim_opt.decay_method,
                        optim_opt.learning_rate, 
                        learning_rate_decay=optim_opt.learning_rate_decay,
                        lr_decay_step=optim_opt.decay_steps,
                        start_decay_steps=optim_opt.start_decay_steps,
                        learning_rate_decay_fn=make_learning_rate_decay_fn(optim_opt), 
                        max_grad_norm=optim_opt.max_grad_norm, 
                        max_weight_value=optim_opt.max_weight_value, 
                        decay_bad_cnt=optim_opt.decay_bad_cnt)
        
        return optimizer
    
    @property
    def training_step(self):
        """The current training step."""
        return self._training_step
    
    def learning_rate(self, better):
        """Returns the current learning rate."""

        if better:
            self._bad_cnt = 0
        else:
            self._bad_cnt += 1

        if self._training_step % self._decay_step == 0 and self._training_step > self._start_decay_steps:
            
            if self._bad_cnt >= self._decay_bad_cnt and self._learning_rate >= 1e-5:
                
                if self._learning_rate_decay_method:
                    scale = self._learning_rate_decay_fn(self._decay_step)
                    self._decay_step += 1
                    self._learning_rate *= scale
                else:
                    self._learning_rate *= self._learning_rate_decay
                
                self._bad_cnt = 0
        
        return self._learning_rate
        
    def state_dict(self):
        return {
            'training_step': self._training_step,
            'decay_step': self._decay_step,
            'optimizer': self._optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self._training_step = state_dict['training_step']
        # State can be partially restored.
        if 'decay_step' in state_dict:
            self._decay_step = state_dict['decay_step']
        if 'optimizer' in state_dict:
            self._optimizer.load_state_dict(state_dict['optimizer'])
    
    def zero_grad(self):
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad()
    
    def backward(self, loss):
        """Wrapper for backward pass. Some optimizer requires ownership of the backward pass."""
        loss.backward()
    
    def step(self):
        """Update the model parameters based on current gradients. """
        learning_rate = self._learning_rate
        for group in self._optimizer.param_groups:
            group['lr'] = learning_rate
            if self._max_grad_norm > 0:
                clip_grad_norm_(group['params'], self._max_grad_norm)
        self._optimizer.step()
        if self._max_weight_value:
            for p in self._params:
                p.data.clamp_(0 - self._max_weight_value, self._max_weight_value)
        self._training_step += 1

    def update_learning_rate(self, better):
        lr0 = self._learning_rate
        lr = self.learning_rate(better)

        if lr != lr0:
            print("Update the learning rate to " + str(lr))
