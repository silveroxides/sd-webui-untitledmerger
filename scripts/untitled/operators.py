import torch,scipy
import scripts.untitled.common as cmn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def recurse(operation):
    source_tensors = []
    for source_oper in operation.sources:
        source_tensor = source_oper.merge()
        source_tensors.append(source_tensor)

    return operation.oper(*source_tensors)

def cache_operation(func):
    def inner(operation):
        try:
            return weights_cache[operation]
        except KeyError:pass

        result = func(operation)

        weights_cache[operation] = result
        return result
    return inner


###OPERATORS####

class Operation:
    def __init__(self,key,*sources):
        self.key = key
        self.sources = tuple(sources)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None
        self.seed = None
        self.merge_func = recurse

    def __eq__(self, other):
        return (self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources) == (other.key, other.alpha, other.beta, other.gamma, other.delta, other.seed, other.sources)
    
    def __hash__(self):
        return hash((self.key, self.alpha, self.beta, self.gamma, self.delta, self.seed, self.sources))
    
    def oper(self,*args) -> torch.Tensor:
        raise NotImplementedError

    def merge(self):
        return self.merge_func(self)
    
    def cache(self):
        if cmn.opts['cache_size'] > 512:
            self.merge_func = cache_operation(recurse)
        return self
        

class LoadTensor(Operation):
    def __init__(self,key,alpha):
        super().__init__(key,*tuple())
        self.alpha = alpha

    #loadtensor uses merge instead of oper as it has no model inputs, use oper everywhere else 
    def merge(self) -> torch.Tensor:
        return cmn.loaded_checkpoints[self.alpha].get_tensor(self.key).to(cmn.device())


class Multiply(Operation):
    def __init__(self,key,alpha,*sources):
        super().__init__(key,*sources)
        self.alpha = alpha

    def oper(self,a) -> torch.Tensor:
        return a * self.alpha


class Add(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    def oper(self,a,b) -> torch.Tensor:
        return a + b


class Sub(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    def oper(self,a,b) -> torch.Tensor:
        return a - b


class Smooth(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    ###From https://github.com/hako-mikan/sd-webui-supermerger
    def oper(self,a) -> torch.Tensor:
        # Apply median filter to the differences
        filtered_diff = scipy.ndimage.median_filter(a.detach().cpu().to(torch.float32).numpy(), size=3)
        # Apply Gaussian filter to the filtered differences
        filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)
        return torch.tensor(filtered_diff,dtype=cmn.dtype(),device=cmn.device())
    

class TrainDiff(Operation):
    def __init__(self,*args):
        super().__init__(*args)

    ###From https://github.com/hako-mikan/sd-webui-supermerger
    def oper(self, a, b, c) -> torch.Tensor:
        if torch.allclose(b.float(), c.float(), rtol=0, atol=0):
            return torch.zeros_like(a)

        diff_AB = b.float() - c.float()

        distance_A0 = torch.abs(b.float() - c.float())
        distance_A1 = torch.abs(b.float() - a.float())

        sum_distances = distance_A0 + distance_A1

        scale = torch.where(sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.).float())
        sign_scale = torch.sign(b.float() - c.float())
        scale = sign_scale * torch.abs(scale)

        new_diff = scale * torch.abs(diff_AB)
        return new_diff.to(cmn.dtype())  *1.8
        

class Extract(Operation):
    def __init__(self,key,alpha,beta,gamma,*args):
        super().__init__(key,*args)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    #From https://github.com/hako-mikan/sd-webui-supermerger
    def oper(self, base: torch.Tensor|None,a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert base is None or base.shape == a.shape
        assert a.shape == b.shape
        assert 0 <= self.alpha <= 1
        assert 0 <= self.beta <= 1
        assert 0 <= self.gamma
        dtype = base.dtype if base is not None else a.dtype
        base = base.float() if base is not None else 0
        a = a.float() - base
        b = b.float() - base
        c = torch.cosine_similarity(a, b, -1).clamp(-1, 1).unsqueeze(-1)
        d = ((c + 1) / 2) ** self.gamma
        result = torch.lerp(a, b, self.alpha) * torch.lerp(d, 1 - d, self.beta)
        return result.to(dtype)
    

class Similarities(Extract):
    def __init__(self,*args):
        super().__init__(*args)

    def oper(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return super().oper(None,a,b)


class PowerUp(Operation):
    def __init__(self,key,alpha, seed, *sources):
        super().__init__(key,*sources)
        self.alpha = alpha
        self.seed = seed

    #https://github.com/martyn/safetensors-merge-supermario/blob/main/merge.py
    #https://arxiv.org/pdf/2311.03099.pdf
    #https://github.com/yule-BUAA/MergeLM/tree/main/model_merging_methods
    def oper(self, a, b):
        # Calculate the delta of the weights
        a, b = resize_tensors(a, b)
        delta = b - a

        # Generate the mask m^t from Bernoulli distribution
        rngenerator = torch.Generator(device=cmn.device())
        rngenerator.manual_seed(self.seed)
        m = torch.empty_like(delta,device=cmn.device(),dtype=cmn.dtype()).uniform_(0,1,generator=rngenerator) < self.alpha

        # Apply the mask to the delta to get δ̃^t
        delta_tilde = m * delta
        
        # Scale the masked delta by the dropout rate to get δ̂^t
        delta_hat = delta_tilde / (1 - self.alpha)
        return delta_hat
    

def resize_tensors(tensor1, tensor2):
    if len(tensor1.shape) not in [1, 2]:
        return tensor1, tensor2

    # Pad along the last dimension (width)
    if tensor1.shape[-1] < tensor2.shape[-1]:
        padding_size = tensor2.shape[-1] - tensor1.shape[-1]
        tensor1 = F.pad(tensor1, (0, padding_size, 0, 0))
    elif tensor2.shape[-1] < tensor1.shape[-1]:
        padding_size = tensor1.shape[-1] - tensor2.shape[-1]
        tensor2 = F.pad(tensor2, (0, padding_size, 0, 0))

    # Pad along the first dimension (height)
    if tensor1.shape[0] < tensor2.shape[0]:
        padding_size = tensor2.shape[0] - tensor1.shape[0]
        tensor1 = F.pad(tensor1, (0, 0, 0, padding_size))
    elif tensor2.shape[0] < tensor1.shape[0]:
        padding_size = tensor1.shape[0] - tensor2.shape[0]
        tensor2 = F.pad(tensor2, (0, 0, 0, padding_size))

    return tensor1, tensor2


class InterpolateDifference(Operation):
    def __init__(self,key,alpha,beta,gamma,seed,*sources):
        super().__init__(key,*sources)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seed = seed

    def oper(self, a, b):
        alpha = max(self.alpha,0.001)

        delta = torch.abs(a - b)

        if self.beta != 1:
            diff = ((torch.max(delta) - delta) / torch.max(delta)) ** (1 / alpha - 1)
        else:
            diff = (delta / torch.max(delta)) ** (1 / alpha - 1)

        diff = torch.nan_to_num(diff)

        rngenerator = torch.Generator(device=diff.device)
        rngenerator.manual_seed(self.seed)
        bitmask = torch.bernoulli(torch.clamp(diff,0,1),out=torch.empty_like(diff),generator=rngenerator)

        interpolated_mask = torch.lerp(bitmask, diff, self.gamma).to(a.dtype)

        res = a * (1 - interpolated_mask) + b * interpolated_mask
        return res

class ManualEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, delta, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha  # Interpolation strength
        self.beta = beta    # Lower threshold for mean differences
        self.gamma = gamma  # Upper threshold for mean differences
        self.delta = delta  # Smoothness factor
        self.seed = seed    # Seed for random number generation

    def oper(self, a, b):
        # Calculate absolute differences
        delta = torch.abs(a - b)
        
        # Normalize differences
        diff = (torch.max(delta) - delta) / torch.max(delta)
        diff = torch.nan_to_num(diff)
        
        # Calculate mean differences
        mean_diff = torch.mean(diff, 0, keepdim=True)
        
        # Create mask based on mean differences
        mask = torch.logical_and(self.beta < mean_diff, mean_diff < self.gamma)
        
        # Apply power function to differences
        powered_diff = diff ** (1 / max(self.alpha, 0.001) - 1)
        powered_diff = torch.nan_to_num(powered_diff)
        
        # Apply mask to powered differences
        masked_diff = powered_diff * mask.float()
        
        # Generate random mask
        rng = torch.Generator(device=a.device)
        rng.manual_seed(self.seed)
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0, 1), generator=rng)
        
        # Interpolate between random mask and powered differences
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.delta)
        
        # Apply final interpolation
        result = a * (1 - interpolated_mask) + b * interpolated_mask
        
        return result.to(a.dtype)

class AutoEnhancedInterpolateDifference(Operation):
    def __init__(self, key, alpha, beta, gamma, seed, *sources):
        super().__init__(key, *sources)
        self.alpha = alpha  # Interpolation strength
        self.beta = beta    # Threshold adjustment factor
        self.gamma = gamma  # Smoothness factor
        self.seed = seed    # Seed for random number generation

    def oper(self, a, b):
        # Calculate absolute differences
        delta = torch.abs(a - b)
        
        # Normalize differences
        max_delta = torch.max(delta)
        diff = (max_delta - delta) / max_delta
        diff = torch.nan_to_num(diff)
        
        # Calculate mean differences
        mean_diff = torch.mean(diff)
        
        # Dynamically set lower and upper thresholds
        lower_threshold = mean_diff * (1 - self.beta)
        upper_threshold = mean_diff * (1 + self.beta)
        
        # Create mask based on dynamic thresholds
        mask = torch.logical_and(lower_threshold < diff, diff < upper_threshold)
        
        # Apply power function to differences
        powered_diff = diff ** (1 / max(self.alpha, 0.001) - 1)
        powered_diff = torch.nan_to_num(powered_diff)
        
        # Apply mask to powered differences
        masked_diff = powered_diff * mask.float()
        
        # Generate random mask
        rng = torch.Generator(device=a.device)
        rng.manual_seed(self.seed)
        random_mask = torch.bernoulli(torch.clamp(masked_diff, 0, 1), generator=rng)
        
        # Interpolate between random mask and powered differences
        interpolated_mask = torch.lerp(random_mask, masked_diff, self.gamma)
        
        # Apply final interpolation
        result = a * (1 - interpolated_mask) + b * interpolated_mask
        
        return result.to(a.dtype)


class WeightSumCutoff(Operation):
    def __init__(self,key,alpha, beta, gamma, *sources):
        super().__init__(key,*sources)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def oper(self, a, b):
        delta = torch.abs(a - b)

        diff = (torch.max(delta) - delta) / torch.max(delta)
        diffn = torch.nan_to_num(diff)

        mean = torch.mean(diffn,0,True) 
        mask = torch.logical_and(mean < self.beta,self.gamma < mean)
        mul = self.alpha*mask

        res = a * (1 - mul) + b * mul
        return res
#The cache
tensor_size = lambda x: x.element_size() * x.nelement()

class WeightsCache:
    def __init__(self, size):
        self.mapping = OrderedDict()
        self.size_cap = min(size, 8192)*1024*1024
        self.size = 0

    def __setitem__(self, key, t):
        if key in self.mapping:
            self.mapping.move_to_end(key)
        else:
            t = t.detach().cpu()
            self.mapping[key] = t
            self.size += tensor_size(t)
            while self.size >= self.size_cap:
                _ , tensor = self.mapping.popitem(last=False)
                self.size -= tensor_size(tensor)

    def __getitem__(self, key: Operation) -> torch.Tensor:
        t = self.mapping[key]
        self.mapping.move_to_end(key)
        return t.clone().to(cmn.device()).type(cmn.dtype())
    

weights_cache = WeightsCache(4096)


