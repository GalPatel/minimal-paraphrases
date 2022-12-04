# Various scoring methods for sentence probabilities using language models

from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import pickle

# prepositions are taken from: https://bitbucket.org/kganes2/text-mining-resources/downloads/
PREPOSITIONS = {'as', 'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'around',
                'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but',
                'by', 'down', 'during', 'except', 'following', 'for', 'from', 'in', 'inside',
                'into', 'like', 'minus', 'minus', 'near', 'next', 'of', 'off', 'on', 'onto', 'onto',
                'opposite', 'out', 'outside', 'over', 'past', 'plus', 'round', 'since', 'since',
                'than', 'through', 'to', 'toward', 'under', 'underneath', 'unlike', 'until', 'up',
                'upon', 'with', 'without'}

TEMPORAL_PREP = {'as', 'aboard','along',#, 'about'
                   'around',
                'at',#'by',
                'during', #'for', 'from',
                'upon', 'with', 'without'}

DETERMINERS = {'a', 'an', 'the'}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("huggingface DEVICE:", device)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_MASK_ID = bert_tokenizer.mask_token_id
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_model.eval()
bert_model = bert_model.to(device)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer_pref = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model.eval()
gpt2_model = gpt2_model.to(device)


# unigram frequencies
# frequincies = dict(pickle.load(open(
#     '/cs/labs/oabend/gal.patel/clause-dataset-generator/data/gpt-openwebtext.pickle', 'rb')))
# total_freq = float(sum(frequincies.values()))

def temporal_preposition(sentence):
    return complete_word(sentence, word_choices=TEMPORAL_PREP)

def complete_word(sentence, word_choices=PREPOSITIONS):
    """
    Completes the masked preposition
    :param sentence: a string representing a sentence, including a '[MASK]' word
    :return: the most likely preposition where [MASK] is
    """
    k = 5  # how many maximal words to pull
    max_place = 0

    input_tokens = bert_tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor(input_tokens).unsqueeze(0)  # Batch size 1
    input_ids = input_ids.to(device)
    outputs = bert_model(input_ids, labels=input_ids)

    loss, prediction_scores = outputs[:2]
    prediction_scores = prediction_scores[0]

    mask_idx = np.argwhere(np.array(input_tokens) == BERT_MASK_ID)[0, 0]  # + 1
    suggested_word = bert_tokenizer.decode([torch.argmax(prediction_scores[mask_idx])])

    topk_predictions = []
    while suggested_word not in word_choices:
        max_place += 1
        if max_place >= prediction_scores.shape[1]:
            return
        if max_place >= len(topk_predictions):
            k = k if max_place < k else min(k * 2,prediction_scores.shape[1])
            topk_predictions = torch.topk(prediction_scores[mask_idx], k=k).indices
        suggested_word = bert_tokenizer.decode([topk_predictions[max_place]])

    # make sure it's a preposition
    # https://www.englishclub.com/vocabulary/prepositions/list.htm

    return suggested_word


def get_perplexity(sentence):
    # https://www.scribendi.ai/can-we-use-bert-as-a-language-model-to-assign-score-of-a-sentence/
    tokenize_input = bert_tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([bert_tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    predictions = bert_model(tensor_input)[0]
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
    return -torch.exp(loss)


def running_mask_prob(sentence):
    """
    Average probability with running mask along the sentence (with BERT)
    """
    tokenize_input = np.array(bert_tokenizer.encode(sentence, add_special_tokens=True))
    # tokenize_input = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))
    all_variations = np.full((len(tokenize_input), len(tokenize_input)), tokenize_input)
    np.fill_diagonal(all_variations, BERT_MASK_ID)
    all_variations = torch.tensor(all_variations, dtype=torch.long)
    all_variations = all_variations.to(device)
    softmax = torch.nn.Softmax(dim=-1)
    predictions = np.array(softmax(bert_model(all_variations)[0]).cpu().data)
    relevant = predictions.diagonal()[tokenize_input].diagonal()[1:-1]
    return np.mean(relevant) * 100



def gpt2_score(sentence):
    # score the sentence probability using gpt2
    input_ids = torch.tensor(
        gpt2_tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(
        0)  # Batch size 1
    num_tokens = input_ids.shape[1]
    sentence_input = input_ids.to(device)
    with torch.no_grad():
        sentence_outputs = gpt2_model(sentence_input, labels=sentence_input)
        sentence_loss = sentence_outputs[0]
        sentence_log_prob = -sentence_loss * num_tokens
    return -sentence_loss  # (1./num_tokens)*(sentence_log_prob-unigram_log_prob)


def unigram_probabilities(sentence):
    tokens = gpt2_tokenizer.tokenize(sentence)
    prob = 1
    for token in tokens:
        prob *= (frequincies[token] / total_freq)
    return prob


def slor(sentence):
    try:
        input_ids = torch.tensor(
            gpt2_tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    except:
        print('cant encode:', sentence)
        return
    num_tokens = input_ids.shape[1]
    sentence_input = input_ids.to(device)
    with torch.no_grad():
        sentence_outputs = gpt2_model(sentence_input, labels=sentence_input)
        sentence_loss = sentence_outputs[0].item()

        sentence_log_prob = -sentence_loss * num_tokens

    words_probs = unigram_probabilities(sentence)
    return (1./num_tokens)*(sentence_log_prob-np.log(words_probs))
