# demo for generating paraphrases

import argparse
from data_paraphraser import paraphrase_segment

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input sentence for paraphrasing')
    return parser.parse_args()

def paraphrase(text):
    segment_para = paraphrase_segment(text)
    results = []
    for sentence_info in segment_para:
        res = {'source':  sentence_info['sentence']}
        if sentence_info['clause']:
            res['clause2nphrase'] = sentence_info['para_cl']
        if sentence_info['active2passive']:
            res['active2passive'] = sentence_info['para_pas']
        results.append(res)
    return results


def demo(text):
    print('input:', text)
    paraphrases = paraphrase(text)
    for para in paraphrases:
        print('###')
        for key, val in para.items():
            print('{}: {}'.format(key, val))


if __name__ == '__main__':

    args = get_args()
    if args.input:
        demo(args.input)

    # paraphrase('She took the book.')