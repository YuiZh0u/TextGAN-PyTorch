import config as cfg
import gensim.models.word2vec as w2v
import torch
from datetime import datetime
from utils.text_process import load_dict, write_tokens, tensor_to_tokens
from math import ceil

def similar_token(samples):

    sample_unique_name = datetime.now().strftime("%Y%m%d_%H%M")
    word2idx_dict, idx2word_dict = load_dict(cfg.dataset)

    print('Algorithm Start: search and replacement by the most similar | {}'.format(sample_unique_name))

    model = w2v.Word2Vec.load('/home/jfrez/AI2/TextGAN-PyTorch/corpusgood_sinrepeticiones.w2v')

    for i, sample in enumerate(samples):
        words_to_mutate = ceil(float(cfg.mutation_rate) * len(torch.nonzero(sample)))
        print('  {}. Mutation rate: {} | Non zero: {} | Words to mutate: {}'.format(i+1, cfg.mutation_rate, len(torch.nonzero(sample)), words_to_mutate))
        for j in range (words_to_mutate): # Modifica los tokens en secuencia
            associated_token = sample[j].item()
            associated_word = idx2word_dict[str(associated_token)]
            try:
                most_similar_raw = model.wv.most_similar(associated_word, topn=1)
                most_similar_word = most_similar_raw[0][0]
                most_similar_pct = most_similar_raw[0][1]
                print('      Original token:word = {}:{} | Similar word = {} | % Similarity = {}'.format(associated_token, associated_word, most_similar_word, most_similar_pct))
                if most_similar_pct >= 0.8:
                    most_similar_token = word2idx_dict[str(most_similar_word)]
                    sample[j] = int(most_similar_token)
                    print('      Token mutado')
            except:
                print('      No existe una palabra similar a {} en el vocabulario'.format(associated_word))
                continue

    mutated_samples = samples
    inp_changed = torch.zeros(samples.size()).long()
    inp_changed[:, 0] = cfg.start_letter
    inp_changed[:, 1:] = mutated_samples[:, :cfg.max_seq_len - 1]

    if cfg.CUDA:
        return inp_changed.cuda(), mutated_samples.cuda()

    return inp_changed, mutated_samples