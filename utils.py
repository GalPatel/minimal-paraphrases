# utils for paraphrasing text. Classes wrap Spacy's classes of doc and token, to enable editing
# the string in a controlled way, while maintaining dependency structure

import spacy
import numpy as np
from lm_probabilities import complete_word, gpt2_score
from spacy.tokens import Token
import warnings
import logging

logging.basicConfig(filename='warnings.log', level=logging.WARNING)
logging.captureWarnings(True)


prp_possesions = {'he': 'his',
                  'she': 'her',
                  'it': 'its',
                  'they': 'their',
                  'i': 'my',
                  'we': 'our',
                  'you': 'your'}


class Word:
    """
    A word in a sentence
    """
    def __init__(self, word, sentence, index, is_punct=False):
        self.sentence = sentence
        self.i = index
        self.text = word
        self.deleted = False
        self.is_punct = is_punct
        self.incl_punct = is_punct or '\'' in word or '\"' in word
        # punctuations = {'\'', '\"', '?', '.', '!', '-', '--'}
        self.left_tight = self.right_tight = False

    def sentence_start(self):
        """
        returns True if the word is the first word (not punctuation token) in the sentence
        """
        if self.i == 0: return True
        for j in range(self.i):
            if not self.sentence[j].is_punct:
                return False
        return True

    def remove(self):
        # delete the word from its sentence
        if self.deleted:
            # print('Already removed \'' + self.text + '\'.')
            return
        self.sentence.remove_word(self.i)
        self.deleted = True

    def replace(self, replacement_text):
        # replace the word with another while keeping its properties (i.e. capitalization and
        # position)
        if self.text[0].isupper() and self.text != 'I' and replacement_text != '[MASK]':
            replacement_text = replacement_text.capitalize()
        word = self.add_after(replacement_text)
        self.remove()
        return word

    def add_after(self, word_text):
        return self.sentence.add_word(word_text, self.i + 1)

    def add_before(self, word_text):
        return self.sentence.add_word(word_text, self.i)

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text


class MPToken(Word):
    """
    Wrapping Spacy's token to enable editing while maintaining dependencies
    """
    def __init__(self, token: spacy.tokens.Token, sentence):
        Word.__init__(self, token.text, sentence, token.i, is_punct=token.is_punct)
        self.dep_ = token.dep_
        self.pos_ = token.pos_
        self.tag_ = token.tag_
        self.lemma_ = token.lemma_
        # self.subtree = token.subtree
        # self.children = token.children
        self.head_spacy_i = token.head.i  # todo safe access
        self.spacy_token = token
        self.spacy_i = token.i
        self.whitespace_ = token.whitespace_
        try:
            self.update_head()
        except:
            pass

    def update_head(self):
        self.head = self.sentence.get_word(self.sentence.spacy2current[self.head_spacy_i])

    def is_leaf(self):
        subtree_idx = [t.i for t in self.subtree()]
        return not len(subtree_idx)

    def right_edge(self):
        subtree_idx = [t.i for t in self.subtree()]
        idx = np.max(subtree_idx)
        return self.sentence.get_word(idx)

    def left_edge(self):
        subtree_idx = [t.i for t in self.subtree()]
        idx = np.min(subtree_idx)
        return self.sentence.get_word(idx)

    def subtree(self):
        if self.sentence.spacy_doc_changed:
            warnings.warn('Returning a subtree for \'' + self.text +
                          '\' but the parent sentence has been modified.')
        return [self.sentence.get_word(self.sentence.spacy2current[word.i]) for word in
                self.spacy_token.subtree if word.i in self.sentence.spacy2current]

    def children(self):
        if self.sentence.spacy_doc_changed:
            warnings.warn('Returning a subtree for \'' + self.text +
                          '\' but the parent sentence has been modified.')
        return [self.sentence.get_word(self.sentence.spacy2current[word.i]) for word in
                self.spacy_token.children if word.i in self.sentence.spacy2current]


class Sentence:
    """
    Wrapping Spacy's doc object
    """
    def __init__(self, doc):
        self.doc = doc  # nlp(sentence)
        self.words = [MPToken(w, self) for w in self.doc]

        self.spacy_doc_changed = False
        self.spacy2current = {token.i: token.i for token in doc}
        self._save_punt_spaces()

        # update head tokens
        for word in self.words: word.update_head()

    def _save_punt_spaces(self):
        prev_whitspace = ''
        for i, word in enumerate(self.words):
            if not word.whitespace_:
                word.right_tight = True
            if not prev_whitspace:
                word.left_tight = True
            prev_whitspace = word.whitespace_

    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def _reduce_text(self):
        text = ''
        if not len(self.words):
            return text
        prev_word = self.words[0]
        text += prev_word.text
        for word in self.words[1:]:
            if (prev_word.incl_punct and prev_word.right_tight) or (
                    word.incl_punct and word.left_tight):
                text += word.text
            else:
                text += ' ' + word.text
            prev_word = word
        return text

    def __repr__(self):
        return self._reduce_text()
    def __str__(self):
        return self._reduce_text()

    def get_word(self, word_idx):
        return self.words[word_idx]

    def remove_word(self, word_idx):
        word_to_remove = self.words[word_idx]
        if type(word_to_remove) == MPToken:
            self.spacy_doc_changed = True
            del self.spacy2current[word_to_remove.spacy_i]
        self.words = self.words[:word_idx] + self.words[word_idx + 1:]
        for index in range(word_idx, len(self.words)):
            word = self.words[index]
            word.i -= 1
            if type(word) == MPToken:
                self.spacy2current[word.spacy_i] = word.i
        del word_to_remove

    def _choose_different_index(self, i):
        # choose an index that if we were to add another word to the sentence, it wouldn't be
        # adjacent to the current i'th word, and neither it would be a start or end of the
        # sentence
        possibilities = list(range(1, i)) + list(range(i + 2, len(self)))
        return np.random.choice(possibilities)

    def _choose_duplicapable_index(self):
        possibilities = [i for i in range(1, len(self)) if not self.get_word(i).is_punct]
        return np.random.choice(possibilities)

    def _dup_separate_tokens(self):
        i, j = np.random.choice(range(1, len(self)), size=2, replace=False)
        new_i, new_j = self._choose_different_index(i), self._choose_different_index(j)
        if new_j > new_i:
            self.dup_word(j, new_j)
            self.dup_word(i, new_i)
        else:
            self.dup_word(i, new_i)
            self.dup_word(j, new_j)

    def _dup_continious_phrase(self):
        dup_possibilities = [i for i in range(0, len(self) - 1) if
                             (not self.get_word(i).is_punct) and
                             (not self.get_word(i + 1).is_punct)]
        i = np.random.choice(dup_possibilities)
        new_possibilities = list(range(1, i)) + list(range(i + 3, len(self)))
        if len(new_possibilities)==0:
            print(str(self))
        new_i = np.random.choice(new_possibilities)
        self.dup_word(i + 1, new_i)
        if new_i <= i:
            self.dup_word(i + 1, new_i)
        else:
            self.dup_word(i, new_i)

    def make_unfluent(self, continious_phrase=True):
        if continious_phrase:
            self._dup_continious_phrase()
        else:
            self._dup_separate_tokens()

    def dup_word(self, word_idx, dup_idx):
        word = self.get_word(word_idx)
        if word_idx == 0 or (word_idx == 1 and self.get_word(0).is_punct):
            dup = self.add_word(word.text.lower(), dup_idx)
        else:
            dup = self.add_word(word.text, dup_idx)
        dup.left_tight = word.left_tight
        dup.right_tight = word.right_tight
        return dup

    def add_word(self, word_text, word_idx):
        self.spacy_doc_changed = True
        if word_idx > len(self.words):
            raise ValueError('Index out of range')
        added_word = Word(word_text, self, word_idx)
        self.words = self.words[:word_idx] + [added_word] + self.words[word_idx:]
        for index in range(word_idx + 1, len(self.words)):
            word = self.words[index]
            word.i += 1
            if type(word) == MPToken:
                self.spacy2current[word.spacy_i] = word.i
        return added_word

    def switch_spans(self, s1, e1, s2, e2):
        if s1 < s2:
            smin, emin = s1, e1
            smax, emax = s2, e2
        else:
            smin, emin = s2, e2
            smax, emax = s1, e1
        if not (smin <= emin and emin < smax and smax <= emax):
            raise ValueError('Spans don\'t compute')
        self.spacy_doc_changed = True
        first_span = [self.get_word(i).text for i in range(smin, emin + 1)]
        second_span = [self.get_word(i).text for i in range(smax, emax + 1)]

        swap_capitalization = False
        if smin == 0:
            swap_capitalization = True
        elif smin == 1:
            # check if there is only a punctuation before smin
            word = self.get_word(0)
            if word.is_punct:
                swap_capitalization = True
        if swap_capitalization:
            second_span[0] = second_span[0].capitalize()
            first_span[0] = first_span[0].lower()

        # insert first span just after second span
        second_pivot = self.get_word(emax)
        for word in first_span:
            second_pivot = second_pivot.add_after(word)
        # remove second span
        for i in range(smax, emax + 1):
            self.get_word(smax).remove()

        # insert second span just after first span
        first_pivot = self.get_word(emin)
        for word in second_span:
            first_pivot = first_pivot.add_after(word)
        # remove first span
        for i in range(smin, emin + 1):
            self.get_word(smin).remove()

        # return the new indices of the spans
        first_start, first_end = smin, smin + emax - smax
        second_start = smax - ((emin - smin) - (emax - smax))
        second_end = second_start + (emin - smin)
        return first_start, first_end, second_start, second_end

    def switch_subtrees(self, i, j):
        s1, e1, s2, e2 = i, i, j, j
        word1, word2 = self.get_word(i), self.get_word(j)
        if type(word1) == MPToken:
            s1, e1 = word1.left_edge().i, word1.right_edge().i
        if type(word2) == MPToken:
            s2, e2 = word2.left_edge().i, word2.right_edge().i
        return self.switch_spans(s1, e1, s2, e2)


TEMPORAL_PREP = {'as', 'aboard', 'along',  # , 'about'
                 'around',
                 'at',  # 'by',
                 'during',  # 'for', 'from',
                 'upon', 'with', 'without'}


def replace_token(doc: Sentence, position, options={}):
    prep = doc.get_word(position)
    mask = prep.replace('[MASK]')
    masked_sentence = str(doc)
    if options:
        bert_rep = complete_word(masked_sentence, options)
    else:
        raise ValueError('No options for token replacement')
    if bert_rep is None:
        return
    if mask.sentence_start():
        bert_rep = bert_rep.capitalize()
    return mask.replace(bert_rep)


class Mask:
    """
    This class manage the process of inserting a preposition/determiner using a masked language model.
    It also controls the choice between with/without this word insertion
    """
    def __init__(self, sentence: Sentence, position, default_word=None, determiner=False):
        self.doc = sentence
        self.position = position
        self.default_word = default_word
        self.basic_sentence = str(self.doc)
        self.sentences = [self.basic_sentence]
        self.score_methods = {'gpt2': gpt2_score}
        self._insert_mask()

    def _insert_mask(self):
        if self.default_word != None:
            prep = self.doc.add_word(self.default_word, self.position)
            self.default_sentence = str(self.doc)
            mask = prep.replace('[MASK]')
        else:
            self.default_sentence = None
            mask = self.doc.add_word('[MASK]', self.position)
        masked_sentence = str(self.doc)
        bert_prep = complete_word(masked_sentence)
        mask.replace(bert_prep)
        self.bert_sentence = str(self.doc)
        self.sentences = [self.basic_sentence, self.bert_sentence, self.default_sentence]
        self._get_scores()

    def _get_scores(self):
        self.default_scores = {s: -float('inf') for s in self.score_methods}
        self.basic_scores = {}
        self.bert_scores = {}
        self.best_sentence = {}
        for method_name in self.score_methods:
            method = self.score_methods[method_name]
            self.basic_scores[method_name] = method(self.basic_sentence)
            self.bert_scores[method_name] = method(self.bert_sentence)
            if self.default_word is not None:
                self.default_scores[method_name] = method(self.bert_sentence)
            arg_best_sent = np.argmax([self.basic_scores[method_name],
                                       self.bert_scores[method_name],
                                       self.default_scores[method_name]])
            self.best_sentence[method_name] = self.sentences[arg_best_sent]

    def get_best_sentence(self, method_name='gpt2'):
        return self.best_sentence[method_name]




class Paraphraser:
    def __init__(self, sentence: Sentence):
        self.doc = sentence
        self.enforced_clause = {}
        self.discarded_clause = {}
        self.discard_string = {}

        self.discard_dep = {}  # something specif or everything not in the other sets
        self.single_dep = {}
        self.multi_dep = {}

        self.deps = self._prepare_deps()

    def _prepare_deps(self):
        # find the clause
        clause = None
        for token in self.doc:
            if token.dep_ == 'advcl':
                clause = token
                break
        if clause is None:
            return
        # check for unwanted clause
        if (clause.lemma_ not in self.enforced_clause) or (clause.lemma_ in self.discarded_clause):
            return

        self.clause = clause
        deps = {}
        for child in clause.children():
            dep = child.dep_
            if dep in self.discard_dep:
                return
            if dep in self.single_dep:
                if dep in deps:
                    return
                deps[dep] = child
            if dep in self.multi_dep:
                if dep in deps:
                    deps[dep].append(child)
                else:
                    deps[dep] = [child]
        return deps
