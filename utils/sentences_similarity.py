import os
import config as cfg
import numpy as np
import re
from math import ceil
from utils.keywords import get_keywords
from gensim.utils import deaccent
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

def cleanText(sentences, single_sentence=False, ignore_num=True):

    puntuacion = '…!"$%&()*+,-./:;<=>?[\]^_`{|}~\''
    if ignore_num:
        puntuacion += '0123456789'
    table = str.maketrans('', '', puntuacion)
    formatted_sentences = []
    for idx, sentence in enumerate(sentences):
        words = []
        if sentence == '':
            continue
        else:
            # sentence_ascii = [sentence.encode('ascii', 'ignore').decode('ascii')] #Solo ascii.
            for sentence_decoded in [sentence.encode('utf8', 'ignore').decode('utf8')]:
                for word in sentence_decoded.split():
                    word = deaccent(word.lower()) #Minusculas + no acentos
                    nopunctuation = [word.translate(table)]
                    word = nopunctuation[0] #Eliminar puntuacion
                    if word == "": #Evitar palabras vacias
                        continue
                    words.append(word)
                new_sentence = "" #Reconstruccion de tweets
                i=0
                for word in words:
                    if i < len(words)-1:
                        i+=1
                        new_sentence = new_sentence + word + " "
                    else:
                        new_sentence = new_sentence + word
                if single_sentence:
                    return new_sentence
                formatted_sentences.append(new_sentence)
    return formatted_sentences


def cleanTextRegex(sentences, single_sentence=False):
    formatted_sentences = []
    for idx, sentence in enumerate(sentences):
        words = []
        if sentence == '':
            continue
        else:
            for sentence_decoded in [sentence.encode('utf8', 'ignore').decode('utf8')]:
                for word in sentence_decoded.split():
                    word = deaccent(word.lower()) #Minusculas + Acentos
                    if word == "": #Evitar palabras vacias
                        continue
                    words.append(word)
                new_sentence = "" #Reconstruccion de tweets
                i=0
                for word in words:
                    if i < len(words)-1:
                        i+=1
                        new_sentence = new_sentence + word + " "
                    else:
                        new_sentence = new_sentence + word

                    clean_sentence = ' '.join(re.sub(r'[^A-Za-z\s]+', '', new_sentence).split())

                if single_sentence:
                    return clean_sentence

                formatted_sentences.append(clean_sentence)
    return formatted_sentences


def sentences_similarity(w2v_model, target_sentence, gpt_sentences):
    sentences_similarity = np.zeros(len(gpt_sentences))

    target_sentence_words = [w for w in target_sentence.split() if w in w2v_model.key_to_index]
    for idx, sentence in enumerate(gpt_sentences):
        if sentence.strip() == '':
            sentences_similarity[idx] = int(0)
        else:
            sentence_words = [w for w in sentence.split() if w in w2v_model.key_to_index]
            sim = w2v_model.n_similarity(target_sentence_words, sentence_words)
            sentences_similarity[idx] = sim

    return sentences_similarity


def mc_similarity(w2v_model, adv_sentence, mc_sentence):
    similarity = 0
    if len(adv_sentence.split()) == 0 or len(mc_sentence.split()) == 0:
        return similarity
    else:
        target_sentence_words = [w for w in adv_sentence.split() if w in w2v_model.key_to_index]
        sentence_words = [w for w in mc_sentence.split() if w in w2v_model.key_to_index]
        if len(target_sentence_words) == 0 or len(sentence_words) == 0:
            return similarity
        else:
            similarity = w2v_model.n_similarity(target_sentence_words, sentence_words)

    return similarity


def getMatchesByWindow(sentence, keywords_list, window_size, first_encounter=False):
    """
    Return n-sized window around the keyword within sentence
    :param sentence: GAN generated sentece
    :param keywords_list: array of keyword to match
    :param window_size: size of words contiguous (window) to return
    :param first_encounter: if True return only the first match; False return all matches in the sentence
    """
    sentence = sentence.split()
    matches = []
    for idx, word in enumerate(sentence):
        if word in keywords_list:
            start = max(0, idx - window_size)
            # print('word={}, idx={}, start={}, total={}'.format(word, idx, start, idx+window_size))
            if first_encounter == True:
                matches = ' '.join(sentence[start:idx+window_size+1])
                return matches
            else:
                matches.append(' '.join(sentence[start:idx+window_size+1]))

    return matches


def getFirstWordsFromSentence(rate, sentence):
    """
    Return first `rate` words from sentence
    :param rate: percentage (in decimal) of words to return
    :param sentence: GAN generated sentece
    """
    formatted_sentence = []
    for i in range(ceil(len(sentence.split())*float(rate))): #Cantidad de palabras a utilizar de base como entrada a GPT2
        word = sentence.split()[i]
        formatted_sentence.append(word)

    return ' '.join(formatted_sentence)


def process_gpt_input(sentences_list, windows_size, rate):
    formatted_seqgan_sentences = []
    keywords = get_keywords(cfg.dataset)
    for sentence in sentences_list:
        if len(sentence.split()) > 0:
            if len(keywords) > 0:
                matches = getMatchesByWindow(sentence, keywords, windows_size, True)
                if len(matches) > 0:
                    formatted_seqgan_sentences.append(matches)
                else:
                    formatted_seqgan_sentences.append(getFirstWordsFromSentence(rate, sentence))
            else:
                formatted_seqgan_sentences.append(getFirstWordsFromSentence(rate, sentence))
        else:
            formatted_seqgan_sentences.append('')
    return formatted_seqgan_sentences


def generate_gpt_sentences(adv_sentences, num_sentences, windows_size, rate):

    final_gpt_sentences = []
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    formatted_seqgan_sentences = process_gpt_input(adv_sentences, windows_size, rate)

    for idx, sentence in enumerate(formatted_seqgan_sentences):
        if sentence == '':
            final_gpt_sentences.append([sentence for i in range(num_sentences)])
        else:
            encoded_input = tokenizer('{} '.format(sentence), return_tensors='pt').input_ids
            outputs = model.generate(
                                    encoded_input,
                                    max_length=cfg.max_seq_len,
                                    num_return_sequences = num_sentences,
                                    pad_token_id=tokenizer.eos_token_id,
                                    do_sample=True,
                                    top_k=50,
                                    top_p=0.95,
                                    temperature = 0.7
                                    )
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            final_gpt_sentences.append(cleanTextRegex(generated_text))

    if cfg.save_reward_samples:
        save_gptsamples_path = os.path.join(cfg.save_root + 'gpt_samples/')
        if not os.path.exists(save_gptsamples_path):
            os.makedirs(save_gptsamples_path)

        with open(save_gptsamples_path + 'sample_{}.txt'.format(datetime.now().strftime("%m%d_%H%M%S")), 'a') as fout:
            fout.write('ADV Sentences:\n  {}\n'.format(str(adv_sentences)))
            fout.write('Formatted Sentences:\n  {}\n'.format(str(formatted_seqgan_sentences)))
            fout.write('Reward Sentences:\n  {}\n'.format(str(final_gpt_sentences)))

    return(np.transpose(final_gpt_sentences))