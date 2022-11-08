# Comming soon to a repo near you :,)
import context
from sknotlearn.optimize import SGradientDescent, GradientDescent
from sknotlearn.neuralnet import NeuralNetwork

def get_opt_params(opt):
    assert type(opt) in (GradientDescent, SGradientDescent), f"{type(opt)} is not a Gradient decent object"
    
    params = {
        "method": opt.method,
    }

    eta1, eta2 = opt.params["eta"](0), opt.params["eta"](100)
    if eta1 == eta2: 
        eta = eta1
    else:
        eta = "varying"

    params["eta"] = eta

    for k, v in opt.params.items():
        if k != "eta":
            params[k] = v

    if type(opt) is GradientDescent:
       params["its"] = opt.its
    else:
        params.update({
            "random_state": opt.random_state,
            "batch_size": opt.batch_size,
            "epochs": opt.epochs
        })

    return params

def get_NN_params(NN):
    assert type(NN) is NeuralNetwork, f"{type(NN)} is not a NeuralNetwork object"

    params = {
        "random_state": NN.random_state,
        "n_hidden_nodes": NN.n_hidden_nodes,
        "n_hidden_layers": NN.n_hidden_layers,
        "lmbda": NN.lmbda
    }

    params.update(NN.func_names)

    return params

def get_all_params(NN):
    d1 = get_NN_params(NN)
    d2 = get_opt_params(NN.optimizer)

    d1["random_state_opt"] = d1.pop("random_state")
    d2["random_state_NN"] = d2.pop("random_state")

    return {**d1, **d2}
