import os
from gensim.models import KeyedVectors, Word2Vec

def get_w2vmodel(dataset):
    abs_dirname = os.path.dirname(os.path.abspath(__file__))

    if dataset == 'covid_tweets':
        coronavirus_CP_path = '/corpusgood_sinrepeticiones.w2v'
        w2vmodel = Word2Vec.load(abs_dirname + coronavirus_CP_path)
        return w2vmodel

    elif dataset == 'hurricane_tweets':
        hurricane_path = '/huracanes_sinrepeticiones.w2v'
        w2vmodel = Word2Vec.load(abs_dirname + hurricane_path)
        return w2vmodel

    elif dataset == 'wikidump':
        wikidump_path = '/wiki_dump_s200_w5_c5_e30.kv'
        w2vmodel = KeyedVectors.load(abs_dirname + wikidump_path)
        return w2vmodel

    else:
        raise NotImplementedError('There is no Word2Vec model for the dataset "{}".'.format(dataset))