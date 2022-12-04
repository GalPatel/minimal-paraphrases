import spacy
nlp = spacy.load("en_core_web_sm")
from utils import Sentence
from clause_nphrase import paraphrase_adverbial_clause
from active_passive import paraphrase_active2passive
import nltk


"""
info/metadata to save about the paraphrases:
'id', 'corpus', 'doc_id', 'seg_id', 'seg_count', 'genre',
'src_lang', 'trg_lang', 'segment', 'sentence', 'sent_id', 'sent_count',
'clause', 'cl_type', 'cl_detector', 'cl_lemma', 'noun_derivation',
'try_prep_cl', 'bert_cl', 'gpt2_cl', 'bert=gpt2_cl', 'para_cl',
'active2passive', 'pas_lemma', 'try_prep_pas', 'bert_pas', 'gpt2_pas', 'bert=gpt2_pas',
'para_pas', 'para_seg', 'translation'
"""



def paraphrase_segment(segment):
    """
    Manage the paraphrasing process for a segment of text. Extract single sentences from the text
    and paraphrase them: clause to noun phrase and/or active to passive (where applicable)
    :param segment: input text (could be multiple sentences)
    :param clause: paraphrase from adverbial clause to noun phrase
    :param active: paraphrase from active to passive
    :return: paraphrased (if applicable) sentence info for each sentence in the text segment (as
    a list)
    """
    # split segment into single sentences:
    sentences = nltk.tokenize.sent_tokenize(segment.strip())
    info_dicts = []
    for i, s in enumerate(sentences):
        info = {}
        sentence_plain_text = str(s) # save original text
        # clause to noun phrase:
        para_info = paraphrase_adverbial_clause(Sentence(nlp(sentence_plain_text)))
        info.update(para_info)
        # active to passive:
        passive_info = paraphrase_active2passive(Sentence(nlp(sentence_plain_text)))
        info.update(passive_info)

        if info['clause'] or info['active2passive']:
            meta = {'sentence': sentence_plain_text, 'sent_id': i + 1, 'sent_count': len(sentences)}
            info.update(meta)
            info_dicts.append(info)


    return info_dicts




