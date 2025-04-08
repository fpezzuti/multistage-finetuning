from typing import Type

from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from mono_encoder_mixin import MonoEncoderMixin, BertMonoEncoderMixin, RoBERTaMonoEncoderMixin, ElectraMonoEncoderMixin


def MonoEncoderClassFactory(TransformerModel: Type[PreTrainedModel]) -> Type[PreTrainedModel]:
    Mixin = get_mixin(TransformerModel)

    assert issubclass(TransformerModel.config_class, PretrainedConfig)
    MonoEncoderConfig = type(
        "MonoEncoderConfig",
        (TransformerModel.config_class,),
        {
            "depth": 100,
            "average_doc_embeddings": False,
            "num_labels": 1,
        },
    )

    def __init__(self, config: PretrainedConfig, use_flash: bool = False, fill_random_docs: bool = True) -> None:
        config.num_labels = 1
        TransformerModel.__init__(self, config)
        Mixin.__init__(self, TransformerModel.forward, use_flash, config.depth if fill_random_docs else None)

    mono_encoder_class = type(
        "MonoEncoderModel",
        (Mixin, TransformerModel),
        {"__init__": __init__, "config_class": MonoEncoderConfig},
    )
    return mono_encoder_class


def get_mixin(TransformerModel: Type[PreTrainedModel]) -> Type[MonoEncoderMixin]:
    if issubclass(TransformerModel, BertPreTrainedModel):
        return BertMonoEncoderMixin
    elif issubclass(TransformerModel, RobertaPreTrainedModel):
        return RoBERTaMonoEncoderMixin
    elif issubclass(TransformerModel, ElectraPreTrainedModel):
        return ElectraMonoEncoderMixin
    else:
        raise ValueError(
            f"Model type {TransformerModel.__name__} not supported by MonoEncoder"
        )
