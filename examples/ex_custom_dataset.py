import torch
import logging
import os
import pickle
from typing import Optional, List

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartClassificationHead

from special_customer import SpecialCustomer

logger = logging.getLogger(__name__)


def load_dataset_from_dir():
    all_codes = []
    print('implementations')
    return all_codes


class CodeDataSet(Dataset):
    def __init__(self):
        super(CodeDataSet, self).__init__()
        self.codes = load_dataset_from_dir()

    def __getitem__(self, index):
        return self.codes[index]

    def __len__(self):
        return self.size

    def save(self):
        path = os.path.join(self.args.dataset_save_dir, f'{self.dataset_name}.pk')
        with open(path, mode='wb') as f:
            pickle.dump(self, f)
        logger.info(f'Dataset saved to {path}')


class BartForClassificationAndGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, mode=None):
        super(BartForClassificationAndGeneration, self).__init__(config)
        print('configurations')
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print('implementations')
        return Seq2SeqLMOutput(
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


def collate_fn(batch):
    print('batch')


class CodeTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super(CodeTrainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset,
                          batch_size=16,
                          shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return DataLoader(dataset=self.eval_dataset,
                          batch_size=16,
                          shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return DataLoader(dataset=self.test_dataset,
                          batch_size=16,
                          shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch))

