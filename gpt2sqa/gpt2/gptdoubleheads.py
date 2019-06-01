
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from gpt2sqa.gpt2.gpt2lmhead import GPT2LMHead, GPT2MultipleChoiceHead
from gpt2sqa.gpt2.gpt2pretrained import GPT2PreTrainedModel
from gpt2sqa.gpt2.gpt2model import GPT2Model


class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    """OpenAI GPT-2 model with a Language Modeling and a Multiple Choice head ("Language Models are Unsupervised Multitask Learners").

    Params:
        config: a GPT2Config class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
            indices selected in the range [0, config.vocab_size[
        `mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
            which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with indices selected in [-1, 0, ..., config.vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., config.vocab_size]
        `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].
        `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
            (key and values in the attention blocks) to speed up sequential decoding
            (this is the presents output of the model, cf. below).

    Outputs:
        if `lm_labels` and `multiple_choice_labels` are not `None`:
            Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
        else: a tuple with
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, config.vocab_size]
            `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]
            `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                torch.FloatTensors. They can be reused to speed up sequential decoding.

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
    mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)

    config = modeling_gpt2.GPT2Config()

    model = modeling_gpt2.GPT2LMHeadModel(config)
    lm_logits, multiple_choice_logits, presents = model(input_ids, mc_token_ids)
    ```
    """

    def __init__(self, config):
        super(GPT2DoubleHeadsModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.multiple_choice_head = GPT2MultipleChoiceHead(config)
        self.apply(self.init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, token_type_ids=None, position_ids=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        losses = []
        if lm_labels is not None:
            shift_logits = lm_logits[:, :-1].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(shift_logits.view(-1,
                                                     shift_logits.size(-1)), shift_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if losses:
            return losses
        return lm_logits, mc_logits, presents
