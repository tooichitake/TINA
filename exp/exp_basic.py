import os
import torch
from models.Autoformer import Model as Autoformer
from models.Transformer import Model as Transformer
from models.TimesNet import Model as TimesNet
from models.Nonstationary_Transformer import Model as Nonstationary_Transformer
from models.DLinear import Model as DLinear
from models.FEDformer import Model as FEDformer
from models.Informer import Model as Informer
from models.LightTS import Model as LightTS
from models.Reformer import Model as Reformer
from models.ETSformer import Model as ETSformer
from models.Pyraformer import Model as Pyraformer
from models.PatchTST import Model as PatchTST
from models.MICN import Model as MICN
from models.Crossformer import Model as Crossformer
from models.FiLM import Model as FiLM
from models.iTransformer import Model as iTransformer
from models.Koopa import Model as Koopa
from models.TiDE import Model as TiDE
from models.FreTS import Model as FreTS
from models.TimeMixer import Model as TimeMixer
from models.TSMixer import Model as TSMixer
from models.SegRNN import Model as SegRNN
from models.MambaSimple import Model as MambaSimple
from models.TemporalFusionTransformer import Model as TemporalFusionTransformer
from models.SCINet import Model as SCINet
from models.PAttn import Model as PAttn
from models.TimeXer import Model as TimeXer
from models.WPMixer import Model as WPMixer
from models.MultiPatchFormer import Model as MultiPatchFormer
from models.KANAD import Model as KANAD
from models.TINA import Model as TINA


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TimesNet": TimesNet,
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "DLinear": DLinear,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "LightTS": LightTS,
            "Reformer": Reformer,
            "ETSformer": ETSformer,
            "PatchTST": PatchTST,
            "Pyraformer": Pyraformer,
            "MICN": MICN,
            "Crossformer": Crossformer,
            "FiLM": FiLM,
            "iTransformer": iTransformer,
            "Koopa": Koopa,
            "TiDE": TiDE,
            "FreTS": FreTS,
            "MambaSimple": MambaSimple,
            "TimeMixer": TimeMixer,
            "TSMixer": TSMixer,
            "SegRNN": SegRNN,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "SCINet": SCINet,
            "PAttn": PAttn,
            "TimeXer": TimeXer,
            "WPMixer": WPMixer,
            "MultiPatchFormer": MultiPatchFormer,
            "KANAD": KANAD,
            "TINA": TINA,
        }
        if args.model == "Mamba":
            print("Please make sure you have successfully installed mamba_ssm")
            from models import Mamba

            self.model_dict["Mamba"] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps")
            print("Use GPU: mps")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
