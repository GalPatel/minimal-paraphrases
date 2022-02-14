# Minimal Paraphrases Pairs
### [Paper](https://arxiv.org/abs/2110.03067)

Minimal Paraphrase Pairs are meaning-preserving paraphrases with a controlled and minimal change.
This repository provides the generation code and a sample dataset derrived from WMT19 dev set.


## Paraphrases Properties
The dataset includes 2 subsets: adverbial clause to noun phrase and active voice to passive voice.

|                               | Source                                      | Paraphrased                                 |
|-------------------------------|---------------------------------------------|---------------------------------------------|
| Active Voice/ Passive Voice   | She **took** the book.                      | The book **was taken** by her.              |
| Adverbial Clause/ Noun Phrase | The party died down before **she arrived**. | The party died down before **her arrival**. |

Sentence pairs attributes:
1. Similar Meaning, for invariant semantics.
2. Minimal Change, to facilitate experimental setup.
3. Controlled Change, where paraphrasing difference is consistent and well-defined. As opposed to 
lexical paraphrases that tend to be idiosyncratic, we require the same distinction to be applied to all instances.
4. Reference Translation, (optional)

The adverbial clauses include: temporal, purpose, cause/reason.

## Sample Dataset
We applied our paraphrases engine on the [English-German WMT19 dev set](https://www.statmt
.org/wmt19/translation-task.html). It can be found in the [data](data) directory.

ome paraphrases, generated automatically by our algorithm, are syntactically well-formed, but also 
anomalous. Therefore, we manually filtered the data provided here.

## Usage for generation of paraphrases
Our algorithm for generating paraphrases will be available here. It can be used to extract 
suitable source sentences and paraphrase them from any given set of English sentences.

## Citation

If you find this dataset or code useful in your works, please acknowledge it
appropriately by citing:

```
@article{DBLP:journals/corr/abs-2110-03067,
  author    = {Gal Patel and
               Leshem Choshen and
               Omri Abend},
  title     = {On Neurons Invariant to Sentence Structural Changes in Neural Machine
               Translation},
  journal   = {CoRR},
  volume    = {abs/2110.03067},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.03067},
  eprinttype = {arXiv},
  eprint    = {2110.03067},
  timestamp = {Thu, 21 Oct 2021 16:20:08 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-03067.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Contact: Gal Patel (gal.patel@mail.huji.ac.il)
