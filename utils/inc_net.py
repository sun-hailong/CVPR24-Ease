import copy
import logging
import torch
from torch import nn
from backbone.linears import CosineLinear
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()    

    elif '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease" :
            from backbone import vit_ease
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_ease":
                model = vit_ease.vit_base_patch16_224_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_ease":
                model = vit_ease.vit_base_patch16_224_in21k_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes):
        self._cur_task += 1
        
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[ : old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
            
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())