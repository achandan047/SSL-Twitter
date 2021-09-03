import numpy as np

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (PretrainedConfig, PreTrainedModel,
                          AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel,
                          BertConfig, BertForSequenceClassification, BertTokenizerFast, BertModel,
                          XLMConfig, XLMForSequenceClassification, XLMTokenizer, XLMModel,
                          XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, XLNetModel,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast, RobertaModel,
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast, DistilBertModel,
                          AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel)



MODEL_CLASSES = {
#     'albert-base': (AlbertConfig, AlbertModel, AlbertTokenizer, 'albert-base-v2'),
#     'albert-large': (AlbertConfig, AlbertModel, AlbertTokenizer, 'albert-large-v2'),
#     'auto-tweet1': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, 'vinai/bertweet-base'),
#     'auto-tweet2': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, 'vinai/bertweet-covid19-base-uncased'),
    'bert-base1': (BertConfig, BertModel, BertTokenizerFast, 'bert-base-uncased'),
#     'bert-base2': (BertConfig, BertModel, BertTokenizer, 'bert-base-uncased'),
#     'bert-large': (BertConfig, BertModel, BertTokenizer, 'bert-large-uncased'),
#     'distilbert-base1': (DistilBertConfig, DistilBertModel, DistilBertTokenizerFast, 'distilbert-base-uncased'),
#     'distilbert-base2': (DistilBertConfig, DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    'roberta-base1': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast, 'roberta-base'),
#     'roberta-base2': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
#     'xlnet-base': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased'),
#     'xlnet-large': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-large-cased')
#     'xlm': (XLMConfig, XLMModel, XLMTokenizer, 'xlm-mlm-en-2048'),
}



class AutoModelRegressionModel(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = 1
        self.num_labels = config.num_labels
        
        super(AutoModelRegressionModel, self).__init__(config)
        self.model = encoder_class.from_config(config)
        self.model_name = model_name

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if labels is not None:
            loss, logits = outputs[:2]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            return (loss, logits)
        else:
            logits = outputs[0]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            return (logits,)
    
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True



class AlbertRegressionModel(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = 1
        self.num_labels = config.num_labels
        
        super(AlbertRegressionModel, self).__init__(config)
        self.model = encoder_class(config)
        self.model_name = model_name
        
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
                logits = torch.clamp(logits, min=-1.0, max=1.0)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)
    
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True

            
            
class BertRegressionModelLite(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = 1
        self.num_labels = config.num_labels
        
        super(BertRegressionModelLite, self).__init__(config)
        self.model = encoder_class(config)
        self.model_name = model_name
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs[1]

        loss_fct = MSELoss()

        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        loss = loss_fct(logits.view(-1), labels.view(-1)) if labels is not None else None
        logits = torch.clamp(logits, min=-1.0, max=1.0)

        return (loss, logits) if loss is not None else (logits,)
    
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True
            
            

class BertRegressionModel(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = 1
        self.num_labels = config.num_labels
        
        super(BertRegressionModel, self).__init__(config)
        self.model = encoder_class(config)
        self.model_name = model_name
        
        # number of samples for multi-sample dropout
        self.samples = 2
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs[1]

        loss_fct = MSELoss()

        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        loss = loss_fct(logits.view(-1), labels.view(-1)) if labels is not None else None
        logits = torch.clamp(logits, min=-1.0, max=1.0)
        
        for _ in range(self.samples - 1):
            output_i = self.dropout(pooled_output)
            logits_i = self.classifier(output_i)
            logits_i = torch.clamp(logits_i, min=-1.0, max=1.0)
            loss_i = loss_fct(logits.view(-1), labels.view(-1)) if labels is not None else None
            
            logits = logits.add(logits_i)
            loss = loss.add(loss_i) if labels is not None else None
        
        logits /= self.samples
        loss = (loss / self.samples) if labels is not None else None

        return (loss, logits) if loss is not None else (logits,)
    
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True



class DistilbertRegressionModel(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = 1
        self.num_labels = config.num_labels
        
        super(DistilbertRegressionModel, self).__init__(config)
        self.model = encoder_class(config)
        self.model_name = model_name
        
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        distilbert_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = torch.clamp(logits, min=-1.0, max=1.0)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)
    
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True
            


class RobertaRegressionModel(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = 1
        self.num_labels = config.num_labels
        
        super(RobertaRegressionModel, self).__init__(config)
        self.model = encoder_class(config)
        self.model_name = model_name

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if labels is not None:
            loss, logits = outputs[:2]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            return (loss, logits)
        else:
            logits = outputs[0]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            return (logits,)
    
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True
            
            

class XLNetRegressionModel(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name, mem_len=1024) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = 1
        
        super(XLNetRegressionModel, self).__init__(config)
        self.model = encoder_class(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if labels is not None:
            loss, logits = outputs[:2]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            return (loss, logits)
        else:
            logits = outputs[0]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            return (logits,)
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True
            
        

# LOCAL_MODEL = {
#     'auto-tweet1': AutoModelRegressionModel,
#     'auto-tweet2': AutoModelRegressionModel,
#     'albert-base': AlbertRegressionModel,
#     'albert-large': AlbertRegressionModel,
#     'bert-base1': BertRegressionModelLite,
#     'bert-base2': BertRegressionModel,
#     'bert-large': BertRegressionModelLite,
#     'distilbert-base1': DistilbertRegressionModel,
#     'distilbert-base2': DistilbertRegressionModel,
#     'roberta-base1': RobertaRegressionModel,
#     'roberta-base2': RobertaRegressionModel,
#     'xlnet-base': XLNetRegressionModel,
#     'xlnet-large': XLNetRegressionModel
# }

LOCAL_MODEL = {
#     'auto-tweet1': AutoModelRegressionModel,
#     'auto-tweet2': AutoModelRegressionModel,
#     'albert-base': AlbertRegressionModel,
#     'albert-large': AlbertRegressionModel,
    'bert-base1': BertRegressionModelLite,
#     'bert-base2': BertRegressionModel,
#     'bert-large': BertRegressionModelLite,
#     'distilbert-base1': DistilBertForSequenceClassification,
#     'distilbert-base2': DistilbertRegressionModel,
    'roberta-base1': RobertaRegressionModel,
#     'roberta-base2': RobertaRegressionModel,
#     'xlnet-base': XLNetRegressionModel,
#     'xlnet-large': XLNetRegressionModel
}
            
            
class SequenceRegressionModel(nn.Module):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, model_path=None):
        super(SequenceRegressionModel, self).__init__()
        encoder_class = LOCAL_MODEL[model_name]
        self.model = encoder_class(model_name, model_path)
        self.config = self.model.config

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
            
            


# MODEL_CLASSES = {
#     'albert-base': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v2'),
#     'albert-large': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, 'albert-large-v2'),
#     'bert-base1': (BertConfig, BertForSequenceClassification, BertTokenizer, 'bert-base-uncased'),
#     'bert-base2': (BertConfig, BertForSequenceClassification, BertTokenizer, 'bert-base-uncased'),
#     'bert-large': (BertConfig, BertForSequenceClassification, BertTokenizer, 'bert-large-uncased'),
#     'distilbert-base': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-uncased'),
# #     'roberta-base': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
#     'xlnet-base': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased')
#     'xlnet-large': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-large-cased')
# #     'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer, 'xlm-mlm-en-2048'),
# }


# class SequenceRegressionModel(PreTrainedModel):
#     """model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.
#     """
#     def __init__(self, model_name, model_path=None):
#         encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
#         config = encoder_config.from_pretrained(options_name) if model_path is None \
#                     else encoder_config.from_pretrained(model_path)
#         config.num_labels = 1
        
#         super(SequenceRegressionModel, self).__init__(config)
#         self.model = encoder_class(config)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
#         if labels is not None:
#             loss, logits = outputs[:2]
#             return loss, logits
#         else:
#             logits = outputs[0]
#             return logits
        
#     def freeze_encoder(self):
#         for param in self.model.parameters():
#             param.requires_grad = False
    
#     def unfreeze_encoder(self):
#         for param in self.model.parameters():
#             param.requires_grad = True