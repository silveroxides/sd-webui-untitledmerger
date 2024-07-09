from typing import Any
import scripts.untitled.operators as opr

CALCMODES_LIST = []

class CalcMode:
    name = 'calcmode'
    description = 'description'
    input_models = 4
    input_sliders = 3

    slid_a_info = '-'
    slid_a_config = (-1, 2, 0.01) #minimum,maximum,step

    slid_b_info = '-'
    slid_b_config = (-1, 2, 0.01)

    slid_c_info = '-'
    slid_c_config = (-1, 2, 0.01)

    slid_d_info = '-'
    slid_d_config = (-1, 2, 0.01)

    def create_recipe(self, key, model_a, model_b, model_c, model_d, seed=False, alpha=0, beta=0, gamma=0, delta=0) -> opr.Operation:
        raise NotImplementedError


class WeightSum(CalcMode):
    name = 'Weight-Sum'
    description = 'model_a * (1 - alpha) + model_b * alpha'
    input_models = 2
    input_sliders = 1
    slid_a_info = "model_a - model_b"
    slid_a_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, **kwargs):
        #This is used when constructing the recipe for the merge, tensors are not handled here.
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)

        if alpha >= 1:
            return b
        elif alpha <= 0:
            return a

        c = opr.Multiply(key, 1-alpha, a)
        d = opr.Multiply(key, alpha, b)
        
        res = opr.Add(key, c, d)
        return res
    
CALCMODES_LIST.append(WeightSum)


class InterpDifference(CalcMode):
    name = 'Comparative Interp'
    description = 'Interpolates between each pair of values from A and B depending on their difference relative to other values'
    input_models = 2
    input_sliders = 3
    slid_a_info = "concave - convex"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "similarity - difference"
    slid_b_config = (0, 1, 1)
    slid_c_info = "binomial - linear"
    slid_c_config = (0, 1, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta = 0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings') or key.startswith('conditioner.embedders.0.transformer.text_model.embeddings') or key.startswith('conditioner.embedders.1.model.token_embedding') or key.startswith('conditioner.embedders.1.model.positional_embedding'):
            return a
        b = opr.LoadTensor(key,model_b)

        return opr.InterpolateDifference(key, alpha, beta, gamma, seed, a ,b)
    
CALCMODES_LIST.append(InterpDifference)


class ManEnhInterpDifference(CalcMode):
    name = 'Enhanced Man Interp'
    description = 'Enchanced interpolation between each pair of values from A and B depending on their difference relative to other values'
    input_models = 2
    input_sliders = 4
    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "lower mean threshold"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "upper mean threshold"
    slid_c_config = (0, 1, 0.001)
    slid_d_info = "smoothness factor"
    slid_d_config = (0, 1, 0.001)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta = 0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key,model_b)

        return opr.ManualEnhancedInterpolateDifference(key, alpha, beta, gamma, delta, seed, a ,b)
    
CALCMODES_LIST.append(ManEnhInterpDifference)


class AutoEnhInterpDifference(CalcMode):
    name = 'Enhanced Auto Interp'
    description = 'Interpolates between each pair of values from A and B depending on their difference relative to other values'
    input_models = 2
    input_sliders = 3
    slid_a_info = "interpolation strength"
    slid_a_config = (0, 1, 0.001)
    slid_b_info = "threshold adjustment factor"
    slid_b_config = (0, 1, 0.001)
    slid_c_info = "smoothness factor"
    slid_c_config = (0, 1, 0.001)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta = 0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        if key.startswith('cond_stage_model.transformer.text_model.embeddings'):
            return a
        b = opr.LoadTensor(key,model_b)

        return opr.AutoEnhancedInterpolateDifference(key, alpha, beta, gamma, seed, a ,b)
    
CALCMODES_LIST.append(AutoEnhInterpDifference)


class AddDifference(CalcMode):
    name = 'Add Difference'
    description = 'model_a + (model_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "addition multiplier"
    slid_a_config = (-1, 2, 0.01)
    slid_b_info = "smooth (slow)"
    slid_b_config = (0, 1, 1)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        diff = opr.Sub(key, b, c)
        if beta == 1:
            diff = opr.Smooth(key,diff)
        diff.cache()

        diffm = opr.Multiply(key, alpha, diff)

        res = opr.Add(key, a, diffm)
        return res
    
CALCMODES_LIST.append(AddDifference)


class TrainDifference(CalcMode):
    name = 'Train Difference'
    description = 'model_a + (model_b - model_c) * alpha'
    input_models = 3
    input_sliders = 1
    slid_a_info = "addition multiplier"
    slid_a_config = (-1, 2, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        diff = opr.TrainDiff(key,a, b, c)
        diff.cache()

        diffm = opr.Multiply(key, alpha, diff)

        res = opr.Add(key, a, diffm)
        return res
    
CALCMODES_LIST.append(TrainDifference)


class Extract(CalcMode):
    name = 'Extract'
    description = 'Adds (dis)similar features between (model_b - model_a) and (model_c - model_a) to model_a'
    input_models = 3
    input_sliders = 4
    
    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)

    slid_b_info = 'similarity - dissimilarity'
    slid_b_config = (0, 1, 0.01)

    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)

    slid_d_info = 'addition multiplier'
    slid_d_config = (-1, 4, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=1, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        extracted = opr.Extract(key, alpha, beta, gamma*15, a, b, c)
        extracted.cache()

        multiplied = opr.Multiply(key, delta, extracted)

        res = opr.Add(key, a, multiplied)
        return res

CALCMODES_LIST.append(Extract)


class AddDisimilarity(CalcMode):
    name = 'Add Dissimilarites'
    description = 'Adds dissimalar features between model_b and model_c to model_a'
    input_models = 3
    input_sliders = 3
    
    slid_a_info = 'model_b - model_c'
    slid_a_config = (0, 1, 0.01)

    slid_b_info = 'addition multiplier'
    slid_b_config = (-1, 4, 0.01)

    slid_c_info = 'similarity bias'
    slid_c_config = (0, 2, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, gamma=0, delta=1, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)
        c = opr.LoadTensor(key,model_c)

        extracted = opr.Similarities(key, alpha, 1, gamma*15, b, c)
        extracted.cache()

        multiplied = opr.Multiply(key, beta, extracted)

        res = opr.Add(key, a, multiplied)
        return res
    
CALCMODES_LIST.append(AddDisimilarity)
        
        
class PowerUp(CalcMode):
    name = 'Power-up (DARE)'
    description = 'Adds the capabilities of model B to model A.'
    input_models = 2
    input_sliders = 2
    slid_a_info = "dropout rate"
    slid_a_config = (0, 1, 0.01)
    slid_b_info = "addition multiplier"
    slid_b_config = (-1, 4, 0.01)

    def create_recipe(key, model_a, model_b, model_c, model_d, alpha=0, beta=0, seed=0, **kwargs):
        a = opr.LoadTensor(key,model_a)
        b = opr.LoadTensor(key,model_b)

        deltahat = opr.PowerUp(key, alpha, seed, a, b)
        deltahat.cache()

        res = opr.Multiply(key, beta, deltahat)

        return opr.Add(key, a, res)

CALCMODES_LIST.append(PowerUp)


