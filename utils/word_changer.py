import config as cfg
import gensim.models.word2vec as w2v
import torch
from datetime import datetime
from utils.text_process import load_dict, write_tokens, tensor_to_tokens

def similar_token(samples):

    word2idx_dict, idx2word_dict = load_dict(cfg.dataset)
    # original_samples = samples
    # sample_unique_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_path_og = cfg.save_samples_root + 'samples_original_{}.txt'.format(sample_unique_name)
    # with open(save_path_og, 'w') as og_samples:
    #     og_samples.write(str(original_samples))

    model = w2v.Word2Vec.load('/home/jfrez/AI2/TextGAN-PyTorch/corpusgood_sinrepeticiones.w2v')

    for i, sample in enumerate(samples):
        associated_token = sample[0].item()
        associated_word = idx2word_dict[str(associated_token)]
        try:
            most_similar = model.wv.most_similar(associated_word)[0][0]
            most_similar_token = word2idx_dict[str(most_similar)]
            print('original token:word = {}:{} - similar word:token = {}:{}'.format(associated_token, associated_word, most_similar, most_similar_token))
            sample[0] = int(most_similar_token)
        except:
            print('No existe una palabra similar a {} en el vocabulario'.format(associated_word))
            continue
        
    mutated_samples = samples
    sample_unique_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path_new = cfg.save_samples_root + 'samples_similar_{}.txt'.format(sample_unique_name)
    write_tokens(save_path_new, mutated_samples)

    inp_changed = torch.zeros(samples.size()).long()
    inp_changed[:, 0] = cfg.start_letter
    inp_changed[:, 1:] = mutated_samples[:, :cfg.max_seq_len - 1]
    
    if cfg.CUDA:
        return inp_changed.cuda(), mutated_samples.cuda()

    return inp_changed, mutated_samples