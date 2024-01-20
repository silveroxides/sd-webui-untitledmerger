from safetensors.torch import safe_open
from safetensors import SafetensorError
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict
import scripts.untitled.operators as oper
import scripts.untitled.misc_util as mutil
import scripts.common as cmn
import torch,os,re,gc
from tqdm import tqdm
from copy import copy,deepcopy
from modules import devices,shared,script_loading,paths,paths_internal

networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

SKIP_KEYS = [
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "betas",
    "log_one_minus_alphas_cumprod",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "posterior_variance",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod"
]


#Items are added at the end of the dict and removed at the beginning 
#High overhead, so is only worth using for computationally demanding operations
class tCache:
    def __init__(self,size):
        self.cache = OrderedDict()
        self.size = size
        self.footprint = 0

    def append(self,key, tensor):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            tensor = tensor.detach().cpu()
            self.cache.update({key:tensor})
            self.footprint += tensor_size(tensor)
            self.clear()

    def retrieve(self,key):
        tensor = self.cache[key]
        self.cache.move_to_end(key)
        return tensor.detach().clone().to(cmn.device).type(cmn.precision)

    def clear(self):
        while self.footprint > self.size:
            tensor = self.cache.popitem(last=False)[1]
            self.footprint -= tensor_size(tensor)

def tensor_size(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()

cmn.tensor_cache = tCache(cmn.cache_size)


def prepare_merge(calcmode,targets,checkpoints,discard_targets,cludes,timer) -> dict:
    cmn.primary = checkpoints[0]
    
    with safe_open_multiple(checkpoints,device=cmn.device) as cmn.loaded_checkpoints:
        state_dict_keys = cmn.loaded_checkpoints[cmn.primary].keys()

        tasks = parse_recipe(calcmode,targets,state_dict_keys,cmn.primary,discard_targets,cludes,checkpoints)
        tasks_copy = copy(tasks)

        state_dict = {}
        #Reuse merged tensors from the last merge's loaded model, if availible
        if shared.sd_model.sd_checkpoint_info.short_title == hash(cmn.last_merge_tasks):
            state_dict,tasks = get_tensors_from_loaded_model(state_dict,tasks)
        
        if cmn.trash_model:
            shared.sd_model.to(device='meta')
            devices.torch_gc()

        timer.record('Prepare merge')
        try:
            with ThreadPoolExecutor(max_workers=cmn.threads) as executor:
                results = executor.map(initialize_merge,tasks)
                executor.shutdown()
        except:
            clear_cache()
            raise
    
    cmn.last_merge_tasks = tuple(tasks_copy)
    state_dict.update(dict(results))
    
    timer.record('Merge')
    return state_dict


def parse_recipe(calcmode,targets,keys,primary,discard,clude,checkpoints) -> list:
    cludemode = clude.pop(0)
    tasks = []

    assigned_keys = assign_weights_to_keys(targets,keys)

    for key in keys:
        #if key in discard_keys:continue
        if key in SKIP_KEYS or 'model_ema' in key:
            tasks.append(oper.LoadTensor(key,primary))
        elif key in assigned_keys.keys():
            tasks.append(calcmode.create_recipe(key,*checkpoints,**assigned_keys[key]))
        else: 
            tasks.append(oper.LoadTensor(key,primary))
    return tasks


def initialize_merge(task) -> tuple:
    try:
        tensor = task.merge()
    except SafetensorError: #Fallback in case one of the secondary models lack a key present in the primary model
        tensor = cmn.loaded_checkpoints[cmn.primary].get_tensor(task.key)

    tensor = tensor.detach().cpu()
    devices.torch_gc()
    #torch.cuda.empty_cache()
    return (task.key, tensor)


def assign_weights_to_keys(targets,keys) -> dict:
    weight_assigners = []
    keystext = "\n".join(keys)+"\n"

    for target_name,weights in targets.items():
        regex = mutil.target_to_regex(target_name)

        weight_assigners.append((weights, regex))
    
    keys_n_weights = list()

    for weights, regex in weight_assigners:
        target_keys = re.findall(regex,keystext,re.M)
        keys_n_weights.append((target_keys,weights))
    
    keys_n_weights.sort(key=lambda x: len(x[0]))
    keys_n_weights.reverse()

    assigned_keys = defaultdict()
    assigned_keys.default_factory = dict
    
    for keys, weights in keys_n_weights:
        for key in keys:
            assigned_keys[key].update(weights)

    return assigned_keys


def get_tensors_from_loaded_model(state_dict,tasks) -> dict:
    intersected = set(cmn.last_merge_tasks).intersection(set(tasks))
    if intersected:
        #clear loras from model
        with torch.no_grad():
            for module in shared.sd_model.modules():
                networks.network_restore_weights_from_backup(module)
        old_state_dict = shared.sd_model.state_dict()

        for task in intersected:
            try:
                state_dict[task.key] = old_state_dict[task.key]
            except:pass
            tasks.remove(task)
        
    return state_dict,tasks


class safe_open_multiple(object):
    def __init__(self,checkpoints,device):
        self.checkpoints = checkpoints
        self.device = device
        self.open_files = {}
     
    def __enter__(self):
        for name in self.checkpoints:
            if name:
                filename = os.path.join(paths_internal.models_path,'Stable-diffusion',name)
                self.open_files[name] = safe_open(filename,framework='pt',device=self.device)
        return self.open_files

    def __exit__(self,*args):
        for file in self.open_files.values():
            file.__exit__(*args)


def clear_cache():
    cmn.tensor_cache.__init__(cmn.cache_size)
    gc.collect()
    devices.torch_gc()
    torch.cuda.empty_cache()
    return "All caches cleared"