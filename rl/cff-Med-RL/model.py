from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
import ray
from typing import Dict, Any, List
import torch
import numpy as np
from transformers import AutoTokenizer

class IMPALATransformer(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, model_name_or_path="bert-base-uncased", num_diseases=256, loss_scale=1.0):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)



        self.config = model_config
        self.encoder_config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

        self.observation_space = obs_space
        self.action_space = action_space

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.1)
        
        self.action_outputs = nn.Linear(self.encoder_config.hidden_size, num_outputs)
        self.value_network = nn.Linear(self.encoder_config.hidden_size, 1)


        self._features = None

    def forward(self, input_dict: Dict, state: List, seq_lens: Any):
        encoded = self.encoder(input_ids = input_dict["obs"].squeeze(1))

        pooled = encoded[1]
        pooled = self.dropout(pooled)
        self._features = pooled

        action_logits = self.action_outputs(pooled)
        
        return action_logits, state
        
    def value_function(self):
        assert self._features is not None, "Must call forward first"
        return torch.reshape(self.activation(self.value_network(self._features)), [-1])
    
class IMPALATransformerLM(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, model_name_or_path="bert-base-uncased", num_diseases=256, loss_scale=1.0):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)



        self.config = model_config
        self.encoder_config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.mlm = BertOnlyMLMHead(self.encoder_config)

        self.observation_space = obs_space
        self.action_space = action_space

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.1)
        self.action_outputs = nn.Linear(self.encoder_config.hidden_size, num_outputs)
        self.value_network = nn.Linear(self.encoder_config.hidden_size, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


        self.loss_fct = torch.nn.CrossEntropyLoss()  
        self.loss_scale = loss_scale
        self._features = None

    def forward(self, input_dict: Dict, state: List, seq_lens: Any):
        encoded = self.encoder(input_ids = input_dict["obs"].squeeze(1))

        # Grab the last hidden state
        pooled = encoded[1]
        pooled = self.dropout(pooled)
        self._features = pooled

        action_logits = self.action_outputs(pooled)
        self.mlm_probability = 0.15
        
        return action_logits, state
        
    def value_function(self):
        assert self._features is not None, "Must call forward first"
        return torch.reshape(self.activation(self.value_network(self._features)), [-1])

    def custom_loss(self, policy_loss, loss_inputs):
        masked, labels = self.torch_mask_tokens(loss_inputs["obs"].squeeze(1))
        attention_mask = loss_inputs["obs"].squeeze(1)!=0
        encoded = self.encoder(masked, attention_mask = attention_mask)
        prediction_scores = self.mlm(encoded[0]) 
        masked_lm_loss = self.loss_fct(prediction_scores.view(-1,len(self.tokenizer)), labels.view(-1))
        self.masked_lm_metric = masked_lm_loss
        
        return [ploss + self.loss_scale * masked_lm_loss for ploss in policy_loss]

    def metrics(self):
        return {
            "mlm_loss": self.masked_lm_metric.cpu().detach()
        }
    def torch_mask_tokens(self, inputs, special_tokens_mask= None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        inputs = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        indices_random = indices_random.to(inputs.device)
        random_words = random_words.to(inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
class IMPALATransformerDiseasePrediction(IMPALATransformer):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, model_name_or_path="bert-base-uncased", num_diseases=256, loss_scale=1.0):
        IMPALATransformer.__init__(self, obs_space, action_space, num_outputs, model_config, name, model_name_or_path="bert-base-uncased", num_diseases=256, loss_scale=1.0)
        
        self.supervised_output = nn.Linear(self.encoder_config.hidden_size, num_diseases)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.loss_scale = loss_scale
        
    def custom_loss(self, policy_loss, loss_inputs):
        supervised_logits = self.activation(self.supervised_output(self._features))
        
        supervised_loss = self.ce_loss(supervised_logits, torch.cuda.LongTensor([ a["disease_target"] for a in loss_inputs["infos"]]))
        self.supervised_loss_metric = supervised_loss
        
        return [ploss + self.loss_scale * supervised_loss for ploss in policy_loss]

    def metrics(self):
        return {
            "supervised_loss": self.supervised_loss_metric.cpu().detach()
        }
        
        
class IMPALAFruitfly(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, K, k, tokenizer_name="bert-base-uncased", model_path=None, fs_path=None):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        nn.Module.__init__(self)

      
        config = AutoConfig.from_pretrained(tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)#Debugging TODO: REMOVE
        self.sample_count = 0 #Debugging TODO: REMOVE
        self.config = model_config
        self.K = K
        self.k = k
        if model_path is not None:
            self.W = torch.load(model_path)
            self.vocab_size = self.W.shape[-1]//2
        else:
            self.vocab_size = config.vocab_size
            self.W = torch.nn.Parameter(torch.randn(K, 2*self.vocab_size, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available else "cpu")), requires_grad=True) 

        self.observation_space = obs_space
        self.action_space = action_space
        self.window_size = 11
        self.e_weight = 0.5# Weight of energy custom loss
        self.activation = nn.GELU()



        self.action_outputs = nn.Linear(K, num_outputs)

        self.value_network = nn.Linear(K, 1)
        if fs_path is not None:
            self.fs = torch.load(fs_path, map_location=self.W.device)
        else:    
            self.fs = torch.nn.Parameter(torch.ones(self.vocab_size*2), requires_grad=False) #Acumulated frequencies of tokens
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict, state: List, seq_lens: Any):
        ids = input_dict["obs"].squeeze(1) 
        
        ids = torch.cat((ids, ids+self.vocab_size), dim=1) # Add the ids to the target part of the sparse vector 
        full_sequence_batch_indices = torch.arange(0, len(ids), dtype=torch.int32).repeat_interleave(input_dict["obs"].shape[-1]*2).to(input_dict['obs'].device)
        full_sequence_coordinates = torch.stack((full_sequence_batch_indices, ids.reshape(-1))).T 
        # Mask out the [PAD] tokens by setting their value to 0, coalesce to shrink the nnzs
        sparse_values = torch.ones_like(full_sequence_coordinates.T[0])*torch.logical_and(ids!=0, ids!=self.vocab_size).reshape(full_sequence_coordinates.T[0].shape)
        out_V_A_s = torch.sparse_coo_tensor(full_sequence_coordinates.T, sparse_values, (ids.shape[0], 2*self.vocab_size), dtype=torch.float32).coalesce()

        with torch.no_grad():
          activations = torch.sparse.mm(out_V_A_s, self.W.T)
          self._features = activations
        binary_hash = torch.zeros_like(activations, dtype=torch.bool)
        order = activations.argsort(dim=1, descending=True)
        trues = order[:,:self.k]
        binary_hash = binary_hash.scatter_(dim=1, index = trues, src=torch.ones_like(trues, dtype=torch.bool))
        logits = self.action_outputs(binary_hash.float())
        return  logits, state

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        ids = loss_inputs["obs"].squeeze(1)

        # Updating the frequency of the tokens
        if self.sample_count < 1e9:
          full_sequence_batch_indices = torch.arange(0, len(ids), dtype=torch.int32).repeat_interleave(loss_inputs["obs"].shape[-1]*2).to(loss_inputs['obs'].device)
          aux_ids = torch.cat((ids, ids+self.vocab_size), dim=1) # Add the ids to the target part of the sparse vector 
          full_sequence_coordinates = torch.stack((full_sequence_batch_indices, aux_ids.reshape(-1))).T 
          out_V_A_s = torch.sparse_coo_tensor(full_sequence_coordinates.T, torch.ones_like(full_sequence_coordinates.T[0]), (ids.shape[0], 2*self.vocab_size), dtype=torch.float32)
          fs = torch.sparse.sum(out_V_A_s, dim=0)
          self.fs[fs.indices().squeeze()] += fs.values()

        # this removes the [PAD] tokens before unfolding, pros better VRAM usage, cons variable batch size
        input_ids = ids[ids!=0]
        input_ids = input_ids.unfold(0, self.window_size, 1).clone().reshape(-1, self.window_size)
        lens = (ids!=0).sum(axis=1) # sequence length for every sample in the batch
        offsets = lens.cumsum(dim=0) # end of each example when filtering out the [PAD] tokens
        offsets_start = offsets - self.window_size + 1 # start of the part of the sequence that is incomplete (or merges 2 sequences) after unfold
        all_rows = torch.arange(len(input_ids) + self.window_size).to(loss_inputs['obs'].device)
        # Now we'll find the rows of the unfolded matrix that are not needed because they merge 2 samples when ignoring the [PAD]
        greater = all_rows.unsqueeze(0) >= offsets_start.unsqueeze(0).T
        lower = all_rows.unsqueeze(0) < offsets.unsqueeze(0).T
        between = torch.logical_and(lower, greater)
        to_keep = torch.logical_not(between.any(dim=0))
        input_ids = input_ids[to_keep[:-self.window_size]]

        input_ids[:,(self.window_size//2)] += self.vocab_size 
        # Get rid of pad target rows
        non_pad = input_ids[:, self.window_size//2] != self.vocab_size 
        input_ids = input_ids[non_pad]
        indices_batch = torch.arange(0,len(input_ids), dtype=torch.int32).repeat_interleave(self.window_size).to(input_ids.device)
        coordinates = torch.stack((indices_batch, input_ids.reshape(-1))).T

        Ps = 1/self.fs[coordinates.T[1]].reshape(input_ids.shape[0], input_ids.shape[1])
        batch_size = input_ids.shape[0] # Batch size in the sense of the windowed sequences
        # Mask out the values of the [PAD] tokens, coalesce the sparse tensor to reduce the number of mms, coalesce() to reduce the nnz hence the multiplications of repeated tokens
        sparse_values = torch.logical_and(input_ids!=0,input_ids!=self.vocab_size).float().reshape(coordinates.T[0].shape)
        V_A_s = torch.sparse_coo_tensor(coordinates.T, sparse_values,(batch_size,2*self.vocab_size),dtype=torch.float32).coalesce()#.to_sparse_csr()

        V_AxWT = torch.sparse.mm(V_A_s, self.W.T)
        mu = V_AxWT.argmax(axis=1)
        W_mu_nonzero = torch.gather(self.W[mu], 1, coordinates.T[1].reshape(batch_size,-1))
        alpha = (W_mu_nonzero*Ps).sum(axis=1)
        denominator = torch.linalg.norm(self.W, dim = 1)[mu]
        E = -(alpha/denominator).sum() 
        self.fruit_fly_energy_metric = E.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])
        
        self.sample_count +=1
        return [(1 - self.e_weight)*loss_+ self.e_weight*E for loss_ in policy_loss]

    @override(TorchModelV2)    
    def value_function(self):
        assert self._features is not None, "Must call forward first"
        return torch.reshape(self.activation(self.value_network(self._features)), [-1])

    def metrics(self):
        return {
                'policy_loss': self.policy_loss_metric,
                'fruit_fly_energy':self.fruit_fly_energy_metric,
                }
