import torch
import config as cfg
from datetime import datetime
from math import ceil
from utils.text_process import load_dict, write_tokens, tensor_to_tokens
from embedding_models.w2v_model import get_w2vmodel

def similar_token(samples, mutation_rate, similar_pct):

    mutated_samples = samples
    sample_unique_name = datetime.now().strftime("%Y%m%d_%H%M")
    word2idx_dict, idx2word_dict = load_dict(cfg.dataset)

    print('Algorithm Start: search and replacement by the most similar | {}'.format(sample_unique_name))

    model = get_w2vmodel(cfg.dataset)

    for i, sample in enumerate(mutated_samples):
        words_to_mutate = ceil(mutation_rate * len(torch.nonzero(sample)))
        print('  {}. Mutation rate: {} | Non zero: {} | Words to mutate: {}'.format(i+1, mutation_rate, len(torch.nonzero(sample)), words_to_mutate))
        for j in range(words_to_mutate): # Modifica los tokens en secuencia
            associated_token = sample[j].item()
            associated_word = idx2word_dict[str(associated_token)]
            try:
                most_similar_raw = model.wv.most_similar(associated_word, topn=1)
                most_similar_word = most_similar_raw[0][0]
                most_similar_pct = most_similar_raw[0][1]
                print('      Original token:word = {}:{} | Similar word = {} | % Similarity = {}'.format(associated_token, associated_word, most_similar_word, most_similar_pct))
                if most_similar_pct >= similar_pct:
                    most_similar_token = word2idx_dict[str(most_similar_word)]
                    sample[j] = int(most_similar_token)
                    print('      Token mutado')
            except:
                print('      No existe una palabra similar a {} en el vocabulario'.format(associated_word))
                continue

    inp_mutated = torch.zeros(samples.size()).long()
    inp_mutated[:, 0] = cfg.start_letter
    inp_mutated[:, 1:] = mutated_samples[:, :cfg.max_seq_len - 1]

    if cfg.CUDA:
        return inp_mutated.cuda(), mutated_samples.cuda()

    return inp_mutated, mutated_samples