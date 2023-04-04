import os

from compressai.zoo import models
from compressai.models import (
    Cheng2020Anchor,
    Cheng2020Attention,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
)
from Network.Basemodel import Basemodel
from Network.FactorGlobalModule import  FactorGlobalModule
from Network.HyperGlobalModule import HyperGlobalModule
from Network.ProposedNetwork import ProposedNetwork
from pathlib import Path
from typing import Dict
models = {
'Cheng2020Anchor':Cheng2020Anchor(128,),
'Cheng2020Attention':Cheng2020Attention(128, ),
'Factor':FactorizedPrior(128, 192),
'Hyper':ScaleHyperprior(128, 192),
'Joint':JointAutoregressiveHierarchicalPriors(192, 192),
"FactorGlobalModule":FactorGlobalModule(),
"HyperGlobalModule":HyperGlobalModule(),
'ProposedBasemodel':Basemodel(),
'Proposed':ProposedNetwork(),
}

import torch

def getmodel(model):
    if model in models:
        return models[model]
    else:
        print("no suitable model")
        exit(1)

    # if models == 'Cheng2020Anchor':
    #     return Cheng2020Anchor(128,)
    # elif models == 'Cheng2020Attention':
    #     return Cheng2020Attention(128, )
    # elif models == 'Factor':
    #     return FactorizedPrior(128, 192)
    # elif models == 'Hyper':
    #     return ScaleHyperprior(128, 192)
    # elif models == 'Joint':
    #     return JointAutoregressiveHierarchicalPriors(192, 192)
    # elif models == "FactorGlobalModule":
    #     return FactorGlobalModule()
    # elif models == "HyperGlobalModule":
    #     return HyperGlobalModule()
    # elif models == 'ProposedBasemodel':
    #     return Basemodel()
    # elif models == 'Proposed':
    #     return ProposedNetwork()
    # else:
    #     print("no suitable model")
    #     exit(1)

def DelfileList(path, filestarts='checkpoint_last'):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(filestarts):
                os.remove(os.path.join(root, file))

def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return state_dict