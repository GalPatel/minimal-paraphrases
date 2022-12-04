from utils import Sentence, prp_possesions, MPToken, Mask, replace_token, TEMPORAL_PREP
from lexicons import lexicon_loader as vd
from clause_detector import detect_advcl
from lm_probabilities import TEMPORAL_PREP

verb2noun, verb_forms = vd.get_verb_noun_derivation(), vd.get_verb_forms()
nomlex = vd.parse_nombank()

def adverbial2noun(verb_token, doc, remove_aux=False):
    """
    Convert verb to noun. Priority to choose AMR's lexicon
    :param verb_token: pointer to the verb token
    :param doc: the sentence/doc the verb is a part of
    :param remove_aux: remove verb's aux if True
    :return: pointer to the changed token and str indicating which lexicon was used
    """
    if remove_aux:
        for child in verb_token.children():
            if child.dep_ in {'aux', 'auxpass'}:
                for token in child.subtree():
                    token.remove()
    verb = verb_token.lemma_.lower()
    if verb in verb2noun: # first priority: AMR lexicon verb to noun
        return verb_token.replace(verb2noun[verb]), 'amr'
    # otherwise, look for possibilities in Nomlex and verb forms
    options = set()
    if verb in nomlex:
        options = options.union(set(nomlex[verb]))
    if verb in verb_forms:
        options.add(verb_forms[verb][vd.PRESENT_PARTICIPLE])
    if len(options) == 0:
        return None, None
    if len(options) == 1:
        noun = verb_token.replace(options.pop())
    else:
        noun = replace_token(doc, verb_token.i, options) # choose best option using LM probability
        if noun is None:
            return None, None
    if verb in verb_forms and verb_forms[verb][vd.PRESENT_PARTICIPLE] == noun.text.lower():
        if len(options) > 1:
            return noun, 'ing>nomlex'
        return noun, 'ing'
    if verb in verb_forms:
        return noun, 'nomlex>ing'
    return noun, 'nomlex'



def insert_possession(subject: MPToken):
    # insert possession
    if subject.tag_ == 'PRP':
        # find it
        subject.replace(prp_possesions[subject.text.lower()])
    else:
        # change it
        subject = subject.right_edge()
        addition = '\''
        if (not subject.tag_ in {'NNPS', 'NNS'}) and (not subject.text.lower().endswith('s')):
            addition += 's'
        subject.add_after(addition)
    return True


def insert_prep(doc: Sentence, derived_noun, default_prep, para_info):
    # insert a preposition following the derived noun.
    para_info['try_prep_cl'] = True
    masking = Mask(doc, derived_noun.i + 1, default_prep)
    para_info['para_cl'] = masking.get_best_sentence('gpt2')  # w/wo prep by gpt2 choice


def paraphrase_reason(doc: Sentence, clause):
    # check for possession
    if clause.lemma_ == 'have':
        return paraphrase_reason_possession(doc, clause)
    return paraphrase_reason_simple(doc, clause)


def paraphrase_reason_possession(doc: Sentence, clause):
    # necessary dependencies:
    deps = {'mark': None, 'dobj': None, 'nsubj': None, 'aux': None, 'neg': None}
    for child in clause.children():
        if child.dep_ == 'mark':
            if child.text.lower() != 'because':
                return
            deps['mark'] = child
        elif child.dep_ in deps:
            if deps[child.dep_] is not None:
                return
            deps[child.dep_] = child
        else:
            return
    if deps['mark'] is None:
        return

    # remove have, remove det of dobj
    for child in deps['dobj'].children():
        if child.dep_ == 'det':
            for token in child.subtree():
                if token.text.lower() == 'no' and deps['neg'] is None:
                    deps['neg'] = token
                else:
                    token.remove()
    clause.remove()

    if deps['aux'] is not None:
        deps['aux'].remove()

    if deps['neg'] is not None:
        # replace aux+neg by lack of
        place = deps['neg'].i
        deps['neg'].remove()
        doc.add_word('of', place)
        doc.add_word('lack', place)

    # because --> because of
    deps['mark'].add_after('of')

    # insert possession
    possession_successful = insert_possession(deps['nsubj'])
    if not possession_successful:
        return

    para_info = {'cl_type': 'cause-reason-possession',
                 'verb2noun': False, 'verb2ing': False,
                 'try_prep_cl': False,
                 'para_cl': str(doc)}
    return para_info


def paraphrase_reason_simple(doc: Sentence, clause):
    # paraphrase reason clause (not possessive form)
    if clause.lemma_ in {'have', 'be', 'do', 'can'}:
        return

    deps = {'mark': None, 'dobj': None, 'subj': None, 'prep': None, 'aux': None}

    for child in clause.children():
        if child.dep_ == 'mark':
            if child.text.lower() != 'because':
                return
            deps['mark'] = child
        elif child.dep_ == 'nsubj' or child.dep_ == 'nsubjpass':
            if deps['subj'] is not None:
                return
            deps['subj'] = child
        elif child.dep_ == 'dobj':
            if deps['dobj'] is not None or deps['prep'] is not None:
                return
            deps['dobj'] = child
        elif child.dep_ == 'prep':
            if deps['dobj'] is not None or deps['prep'] is not None:
                return
            deps['prep'] = child
        elif child.dep_ in {'aux', 'auxpass', 'npadvmod', 'punct', 'agent', 'attr'}:
            continue
        else:
            return
    if deps['mark'] is None or deps['subj'] is None:
        return

    # because --> because of
    deps['mark'].add_after('of')

    possession_successful = insert_possession(deps['subj'])
    if not possession_successful:
        return

    derived_noun, noun_method = adverbial2noun(clause, doc, remove_aux=True)
    if derived_noun is None:
        return
    para_info = {'cl_type': 'cause-reason-simple',
                 'noun_derivation': noun_method,
                 'try_prep_cl': False}

        # if there is dobj, add 'of' or another preposition
    para_info['para_cl'] = str(doc)
    if deps['dobj'] is not None:
        # if its of the form verb + xxxself, change to self + noun derivation
        if deps['dobj'].text.lower().endswith('self'):
            deps['dobj'].remove()
            derived_noun.add_before('self')
            para_info['para_cl'] = str(doc)
        else:
            insert_prep(doc, derived_noun, 'of', para_info)
    return para_info


def paraphrase_purpose(doc: Sentence, clause):
    deps = {'part': None, 'dobj': None, 'nsubj': None}
    for child in clause.children():
        if child.dep_ == 'aux' and child.tag_ == 'TO' and child.text.lower() == 'to':
            if deps['part'] is not None:
                return
            deps['part'] = child
        elif child.dep_ == 'nsubj':  # or child.dep_ == 'nsubjpass':
            if deps['nsubj'] is not None:
                return
            deps['nsubj'] = child
        elif child.dep_ == 'dobj':
            if deps['dobj'] is not None:
                return
            deps['dobj'] = child
    if deps['part'] is None:  # or deps['subj'] is None:
        return

    deps['part'].replace('for')
    derived_noun, derivation_method = adverbial2noun(clause, doc)
    if derived_noun is None:
        return
    # insert possession to nsubj:
    para_info = {'cl_type': 'purpose',
                 'noun_derivation': derivation_method,
                 'try_prep_cl': False,
                 'para_cl': str(doc)}
    if deps['dobj'] is not None:
        insert_prep(doc, derived_noun, 'of', para_info)
    return para_info


def paraphrase_temporal(doc: Sentence, clause):
    if clause.lemma_ in {'have', 'be'}:
        return
    if 'as soon as' in str(doc).lower(): # difficult case, discard
        return

    deps = {'when': None, 'mark': None, 'dobj': None, 'nsubj': None}
    for child in clause.children():
        if child.dep_ == 'advmod' and child.text.lower() == 'when':
            if deps['when'] is not None:
                return
            deps['when'] = child
        elif child.dep_ == 'mark' and child.text.lower() in {'as', 'before', 'after', 'until',
                                                             'while'}:
            if deps['mark'] is not None:
                return
            deps['mark'] = child
        elif child.dep_ == 'nsubj':  # or child.dep_ == 'nsubjpass':
            if deps['nsubj'] is not None:
                return
            deps['nsubj'] = child
        elif child.dep_ == 'dobj':
            if deps['dobj'] is not None:  # or deps['prep'] is not None:
                return
            deps['dobj'] = child
        elif child.dep_ == 'advmod':
            return
    if deps['when'] is None and deps['mark'] is None:
        return
    if deps['when'] is not None and deps['mark'] is not None:
        return
    if deps['nsubj'] is None:
        return

    time_prep = None
    # when -> upon
    if deps['when'] is not None:
        time_prep = deps['when'].replace('upon')

    # as -> during
    if deps['mark'] is not None:
        if deps['mark'].text.lower() in {'as', 'while'}:
            time_prep = deps['mark'].replace('during')

    # insert possession to nsubj:
    possession_successful = insert_possession(deps['nsubj'])
    if not possession_successful:
        return

    if time_prep is not None:
        replace_token(doc, time_prep.i, TEMPORAL_PREP)

    derived_noun, derivation_method = adverbial2noun(clause, doc, remove_aux=True)
    if derived_noun is None:
        return
    para_info = {'cl_type': 'temporal',
                 'noun_derivation': derivation_method,
                 'try_prep_cl': False,
                 'para_cl': str(doc)}
    if deps['dobj'] is not None:
        insert_prep(doc, derived_noun, 'of', para_info)

    return para_info


def paraphrase_adverbial_clause(sentence: Sentence):
    # paraphrase clause to noun phrase (if applicable)
    # return meta data info about the paraphrase
    detected_type, detector, advcl = detect_advcl(sentence)
    para_info = {'clause': False} # default - clause paraphrasing was not applicable
    para_funcs = {'temporal': paraphrase_temporal,
                  'purpose': paraphrase_purpose,
                  'cause-reason': paraphrase_reason}
    if detected_type in para_funcs:
        paraphrase = para_funcs[detected_type](sentence, advcl)
        if paraphrase is not None:
            para_info.update(paraphrase)
            para_info['clause'] = True
            para_info['cl_lemma'] = advcl.lemma_
            para_info['cl_detector'] = detector
    return para_info
