import spacy
import os
from utils import Sentence, Mask
from lexicons import lexicon_loader as vd


nlp = spacy.load("en_core_web_sm")
verb_forms = vd.get_verb_forms()

# proper nouns conversions
prp_subject2object = {'i': 'me', 'you': 'you', 'we': 'us', 'he': 'him', 'she': 'her',
                          'they': 'them', 'it': 'it'}
prp_object2subject = {'me': 'I', 'you': 'you', 'us': 'we', 'him': 'he', 'her': 'she',
                          'them': 'they', 'it': 'it'}
aux_verbs_convertion = {
        'will': 'will',
        'can': 'could',
        'could': 'could',
        'may': 'might',
        'might': 'might',
        'shall': 'should',
        'should': 'should'
    }
def paraphrase_active2passive(sentence):
    paraphrase = active2passive(sentence)
    if paraphrase:
        return paraphrase
    return {'active2passive': False}


def active2passive(sentence, return_dict=True):
    """
    Convert active voice sentence to passive voice - only if applicable, i.e. if source sentence
    is detected as active voice, not a question or a coordination sentence,
    :param sentence: input str sentence or a Sentence object
    :param return_dict: wether to return a dictionary with meta info about the paraphrasing
    :return: a converted sentence (or info dict) if paraphrasing was succeful. Otherwise,
    returns None
    """
    if type(sentence)==str:
        doc = Sentence(nlp(sentence))
    else:
        doc = sentence
    if str(sentence).endswith('?'): # discard question sentences
        return
    deps = {'nsubj': None, 'ROOT': None, 'dobj': None, 'aux': None, 'dative': None, 'prt': None,
            'neg': None, 'advmod': None}
    # save quick access to useful components
    for token in doc:
        if token.dep_ == 'ROOT':
            deps['ROOT'] = token
            continue
        if token.head.dep_ == 'ROOT' and token.dep_ in deps and (deps[token.dep_] is None):
            deps[token.dep_] = token
            continue

        if token.tag_ == 'WRB' or token.tag_ == 'WP':  # question
            return
        if token.dep_ == 'cc':  # coordination
            return
    for d in ['nsubj', 'ROOT', 'dobj']:
        if deps[d] == None:
            return
    if deps['ROOT'].tag_ == 'VBN' and deps['aux'] is not None:
        return
    if deps['aux'] is not None and deps['aux'].text.lower() == 'to':
        return

    # singular/plural, person, tense
    number = person = tense = None
    if deps['nsubj'].tag_ == 'PRP':
        if not deps['nsubj'].text.lower() in prp_subject2object:
            return
        deps['nsubj'].replace(prp_subject2object[deps['nsubj'].text.lower()])
    if deps['dobj'].tag_ == 'PRP':
        if deps['dobj'].text.lower() not in prp_object2subject:
            return
        new_subject = prp_object2subject[deps['dobj'].text.lower()]
        deps['dobj'].replace(new_subject)
        if new_subject in {'i', 'we'}:
            person = 'first'
        elif new_subject in {'you'}:
            person = 'second'
        elif new_subject in {'he', 'she', 'it', 'they'}:
            person = 'third'
        if new_subject in {'i', 'he', 'she', 'it'}:
            number = 'singular'
        else:
            number = 'plural'
    else:
        if deps['dobj'].tag_ in {'NN', 'NNP'}:
            number = 'singular'
        elif deps['dobj'].tag_ in {'NNS', 'NNPS'}:
            number = 'plural'
        else:
            RuntimeWarning('Object \'' + deps['dobj'].text + '\' without number, got tag ' + deps[
                'dobj'].tag_)

    # # look for advmods
    # advmods = []
    # for child in deps['dobj'].children():
    #     if child.dep_ == 'advmod':
    #         advmods.append(child)

    # switch spans of subject and dobj
    subj_start, subj_end, obj_start, obj_end = doc.switch_subtrees(deps['dobj'].i, deps['nsubj'].i)
    doc.add_word('by', obj_start)

    aux = None

    if deps['aux'] is not None:
        aux = deps['aux']
        if deps['aux'].lemma_.lower() in aux_verbs_convertion:
            deps['aux'].add_after('be')
            deps['aux'].replace(aux_verbs_convertion[deps['aux'].lemma_.lower()])
            tense = 'future'
        else:
            if deps['aux'].tag_ == 'VBD':
                tense = 'past'
            elif deps['aux'].tag_ == 'VBP':
                tense = 'present'

            if deps['ROOT'].tag_ == 'VBG':
                deps['aux'].replace(verb_forms['be'][vd.PRESENT_PARTICIPLE])
            else:
                deps['aux'].remove()

    else:
        # detect tense:
        if deps['ROOT'].tag_ in {'VBG', 'VBP', 'VBZ', 'VB'}:  # todo VB
            tense = 'present'
        elif deps['ROOT'].tag_ in {'VBD', 'VBN'}:
            tense = 'past'
        elif deps['ROOT'].tag_ == 'VB':
            return
    if deps['ROOT'].lemma_.lower() not in verb_forms:
        return

    if tense != 'future':
        if tense == 'past':
            if number == 'singular':
                new_aux = 'was'
            else:
                new_aux = 'were'
        else:
            if number == 'plural':
                new_aux = 'are'
            else:
                if person is not None:
                    if person == 'first':
                        new_aux = 'am'
                    else:
                        new_aux = 'is'
                else:
                    new_aux = 'is'
        aux = doc.add_word(new_aux, subj_end + 1)

    if deps['neg'] is not None:
        deps['neg'].remove()
        aux.add_after('not')
    root_lemma = deps['ROOT'].lemma_.lower()
    placeholder = deps['ROOT'].replace(verb_forms[root_lemma][vd.PAST_PARTICIPLE])
    if deps['prt'] is not None:
        prt = deps['prt'].text
        deps['prt'].remove()
        # placeholder = placeholder.add_after(prt)

    info = {'active2passive': True, 'pas_lemma': root_lemma}
    if deps['dative'] is not None:
        if return_dict:
            info['try_prep_pas'] = True
            masking = Mask(doc, deps['dative'].left_edge().i)
            # info['bert_pas'] = masking.get_best_sentence('bert')
            info['gpt2_pas'] = masking.get_best_sentence('gpt2')
            # info['bert=gpt2_pas'] = info['bert_pas'] == info['gpt2_pas']
            info['para_pas'] = info['gpt2_pas']
        else:
            # masking = Mask(doc, deps['dative'].i)
            return #'DATIVE\n' + masking.get_all_options()
    if return_dict:
        info['para_pas'] = str(doc)
        return info
    return #str(doc)


