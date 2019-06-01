from torch import nn


class GPT2LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2MultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config):
        super(GPT2MultipleChoiceHead, self).__init__()
        self.n_embd = config.n_embd
        self.linear = nn.Linear(config.n_embd, 1)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        # Classification logits
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
        # mc_token_ids (bsz, num_choices)
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        # (bsz, num_choices, 1, hidden_size)
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        # (bsz, num_choices, hidden_size)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        # (bsz, num_choices)
        return multiple_choice_logits
