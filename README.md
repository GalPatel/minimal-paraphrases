# Minimal Paraphrases Pairs
### [Paper](https://arxiv.org/abs/2110.03067)

Minimal Paraphrase Pairs are meaning-preserving paraphrases with a controlled and minimal change.
This repository provides the script to generate new paraphrases and a sample dataset derrived from WMT19 English-German dev set.


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
We applied our paraphrases engine on the English-German [WMT19](https://www.statmt.org/wmt19/translation-task.html) dev set. It can be found in the [data](data) directory.

Some paraphrases, generated automatically by our algorithm, are syntactically well-formed, but also 
anomalous. Therefore, we manually filtered the data provided here.

## Usage for generation of paraphrases
You can generate your own paraphrases with our script (install `requirements.txt`)

* You can paraphrase single sentences or entire paragraphs.
* Each sentence would be paraphrased into both passive voice and noun-phrase (if applicable)

A quick demo:
```
python demo.py "She took the book before she left the house. She went to the local caffe because she loves their pie."
```
Output:
```
input: She took the book before she left the house. She went to the local caffe because she loves their pie.
###
source: She took the book before she left the house.
clause2nphrase: She took the book before her departure from the house.
active2passive: The book was taken by her before she left the house.
###
source: She went to the local caffe because she loves their pie.
clause2nphrase: She went to the local caffe because of her love for their pie.
```

#### Notes:
1. For easy code usage, see `paraphrase` in `demo.py`
2. The script automatically uses gpu if detected



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
