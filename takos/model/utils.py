from takos.model.sequence_tagger.bilstm_crf import BilstmCRF
from takos.model.sequence_tagger.cnn_bilstm_crf import CNNBilstmCRF
from takos.model.sequence_tagger.transformer import TransformerTagger
from takos.model.sequence_tagger.transformer import TransformerCRF
from takos.model.sequence_tagger.bert import BertTagger


from pytorch_pretrained_bert.modeling import BertConfig


def create_model(type, tag_to_idx, model_configs):
    model_type = model_configs['type']
    model_params = model_configs['parameters']
    if type == 'ner' or type == 'word_segment':
        vocab_size = model_configs['vocab_size']
        if model_type == 'bilstm_crf':
            model = BilstmCRF(vocab_size, tag_to_idx, model_params['word_embedding_dims'],
                              model_params['hidden_dims'])
        elif model_type == 'cnn_bilstm_crf':
            model = CNNBilstmCRF(vocab_size, tag_to_idx, model_params['word_embedding_dims'],
                                 model_params['channel_dims'], model_params['conv_configs'],
                                 model_params['hidden_dims'])
        elif model_type == 'transformer':
            model = TransformerTagger(vocab_size, len(tag_to_idx), model_params['word_embedding_dims'],
                                      model_params['hidden_dims'], model_params['head_size'],
                                      model_params['layer_size'])
        elif model_type == 'transformer_crf':
            model = TransformerCRF(vocab_size, tag_to_idx, model_params['word_embedding_dims'],
                                   model_params['hidden_dims'], model_params['head_size'],
                                   model_params['layer_size'])
        elif model_type == 'bert':
            bert_configs = BertConfig(model_params['config_path'])
            model = BertTagger(bert_configs, len(tag_to_idx))
        else:
            raise ValueError()
    else:
        raise ValueError()

    return model
