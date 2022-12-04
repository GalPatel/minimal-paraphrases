# adverbial clause detector - find an adverbial clause and its type
import spacy
from collections import defaultdict
# from allennlp.predictors import Predictor
from allennlp_models import pretrained
from utils import Sentence, MPToken
import torch


cuda_device = 0 if torch.cuda.is_available() else -1
# PREDICTOR = Predictor.from_path("srl-model-2018.05.25.tar.gz", cuda_device=cuda_device)
PREDICTOR = pretrained.load_predictor('structured-prediction-srl-bert')
print('cuda_device for SRL predictor:', cuda_device)
nlp = spacy.load("en_core_web_sm")

TEMPORAL_WORDS = {'when', 'whenever', 'before', 'after', 'while', 'until',
                  'since'}  # 'as soon as', 'as'
PLACE_WORDS = {'where', 'wherever', 'everywhere'}
CONDITION_WORDS = {'if', 'unless'}
MANNER_AS_WORDS = {'if', 'though'}
MANNER_SINGLE_WORDS = {'like', 'as'}
CAUSAL_WORDS = {'because', 'since'}  # 'as'
COMPARISON_WORDS = {'than'}
CONCESSION_CONTRAST_WORDS = {'although', 'though', 'while', 'whereas'}
PAIRED_EVEN_CONCESSION_WORDS = {'if', 'though', 'while'}
PURPOSE_WORDS = {'lest', 'that'}
types = {'temporal', 'cause-reason', 'place', 'manner', 'condition', 'comparison',
         'concession-contrast', 'purpose', 'participle', 'NA'}





def adverbial_clause_extractor(sentence):
    # get a pointer to the root of adverbial clause inside the sentence
    # if not found - return none
    if type(sentence) == str:
        doc = nlp(sentence)
    else:
        doc = sentence
    # filtering is not by length but by advcl complexity
    # look for sentences with a single adverbial clause (and none other!)
    advcl = None

    for token in doc:
        # unwanted clauses - not an adverbial clause
        unwanted_clauses = {'acl', 'csubj', 'csubjpass', 'ccomp', 'xcomp', 'relcl'}
        if token.dep_ in unwanted_clauses:
            return
        if token.dep_ == 'conj' and token.pos_ == 'VERB':
            return
        if token.dep_ == 'advcl' and token.head.dep_ == 'ROOT':
            if advcl is not None:
                return
            advcl = token

    return advcl


def convert_srl2type(srl):
    """
    Convert AllenNLP's SRL names to readable names
    """
    if srl.endswith('TMP'):
        return 'temporal'
    if srl.endswith('MNR'):
        return 'manner'
    if srl.endswith('CAU'):
        return 'cause-reason'
    if srl.endswith('PRP'):
        return 'purpose'
    return srl[-3:]


def detect_clause_srl(clause, sentence):
    """
    Detect clause type using AllenNLP's SRL
    :param clause: clause subtree pointer
    :param sentence: sentence object
    :return: name of clause type
    """
    results = PREDICTOR.predict(sentence=sentence)
    for verb in results['verbs']:
        if 'ARGM' in verb['tags'][clause.i]:
            return convert_srl2type(verb['tags'][clause.i])
        if verb['verb'] == clause.text and verb['tags'][clause.i] == 'B-V':
            for tag in verb['tags']:
                if 'ARGM' in tag:
                    return convert_srl2type(tag)
    # return 'NA'


def detect_advcl_type(clause):
    """
    Detect type of clause (e.g. temporal) by heuristic rules
    """
    subree_pos = defaultdict(lambda: [])
    subree_dep = defaultdict(lambda: [])

    subtree = clause.subtree() if type(clause) == MPToken else clause.subtree
    for t in subtree:
        subree_pos[t.pos_].append(t)
        subree_dep[t.dep_].append(t)

    if len(subree_dep['advmod']) > 0:
        for t in subree_dep['advmod']:
            if t.head == clause:
                text = t.text.lower()
                if text in TEMPORAL_WORDS:  # temporal
                    return 'temporal'
                elif text in PLACE_WORDS:  # location / place
                    return 'place'

    if len(subree_dep['mark']) > 0:
        if len(subree_dep['mark']) == 1:
            t = subree_dep['mark'][0]
            if t.head == clause or (t.head.pos_ == 'AUX' and t.head.head == clause):
                text = t.text.lower()
                if text in CONDITION_WORDS:
                    return 'condition'
                elif text in MANNER_SINGLE_WORDS:
                    return 'manner'
                elif text == 'as':
                    return 'temporal'
                elif text in CAUSAL_WORDS:
                    return 'cause-reason'
                elif text in COMPARISON_WORDS:
                    return 'comparison'
                elif text in CONCESSION_CONTRAST_WORDS:
                    return 'concession-contrast'
                elif text in PAIRED_EVEN_CONCESSION_WORDS:
                    if 'even' in set(a.text.lower() for a in subree_dep['advmod']):
                        return 'concession-contrast'
                elif text in PURPOSE_WORDS:
                    return 'purpose'
        elif len(subree_dep['mark']) == 2:
            for i, t in enumerate(subree_dep['mark']):
                if t.text.lower() == 'as' and len(subree_dep['mark']) - 1 >= (i + 1 % 2) and \
                        subree_dep['mark'][i + 1 % 2].text.lower() in MANNER_AS_WORDS:
                    return 'manner'
                elif t.text.lower() == 'so' and len(subree_dep['mark']) - 1 >= (i + 1 % 2) and \
                        subree_dep['mark'][i + 1 % 2].text.lower() == 'that':
                    return 'purpose'
    if clause.tag_ == 'VB':
        for t in subree_dep['aux']:
            if t.text.lower() == 'to':
                return 'purpose'
    if clause.tag_ in {'VBG', 'VBN'}:
        return 'participle'
    return 'NA'




def detect_advcl(sentence: Sentence):
    """
    Detect adverbial clause and its type (e.g. temporal) in a sentence
    :param sentence: Sentence object
    :return: detected clause type, detection method (AllenNLP SRL or heuristic), the clause pointer
    """
    advcl = adverbial_clause_extractor(sentence)
    detector = detected_type = None
    if advcl != None:
        detected_type = detect_clause_srl(advcl, str(sentence))
        if detected_type in types:
            detector = 'allennlp\'s SRL'
        else:
            detected_type = detect_advcl_type(advcl)
            detector = 'heuristic'
    return detected_type, detector, advcl