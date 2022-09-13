# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : rollout.py
# @Time         : Created at 2019-03-15
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import copy
import torch
import torch.nn.functional as F

import numpy as np
from embedding_models.w2v_model import get_w2vmodel
from utils.sentences_similarity import mc_similarity, generate_gpt_sentences

import config as cfg
from utils.text_process import load_dict, write_tokens, tensor_to_tokens

class ROLLOUT:
    def __init__(self, gen, gpu=True):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_seq_len
        self.vocab_size = gen.vocab_size
        self.step_size = gen.step_size if gen.name == 'leakgan' else 0
        self.goal_out_size = gen.goal_out_size if gen.name == 'leakgan' else 0
        self.gpu = gpu
        self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        # for i in range(given_num):
        inp = sentences[:, :given_num]
        out, hidden = self.gen.forward(inp, hidden, need_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if self.gpu:
            samples = samples.cuda()

        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

            out, hidden = self.gen.forward(inp, hidden, need_hidden=True)

        return samples


    def get_reward(self, sentences, rollout_num, dis, current_k=0):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen
        :return: reward: [batch_size]
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()

            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(sentences, given_num)
                    out = dis.forward(samples)
                    out = F.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1

        # rewards = torch.mean(rewards, dim=0)
        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        return rewards


    def get_reward_similarity(self, sentences, rollout_num):

        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards_np = np.zeros([rollout_num * self.max_seq_len, batch_size])

            adv_sentences = []
            for array_ofwords in tensor_to_tokens(sentences, self.idx2word_dict):
                  adv_sentences.append(' '.join(array_ofwords))
            w2v_model = get_w2vmodel('wikidump') #TODO: que sea cfg y se cargue desde la funcion padre

            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(sentences, given_num)
                    mc_tokens = tensor_to_tokens(samples, self.idx2word_dict)
                    mc_sentences = []
                    for arr_of_word in mc_tokens:
                        mc_sentences.append(' '.join(arr_of_word))

                    for i in range(batch_size):
                        rewards_np[idx][i] = mc_similarity(w2v_model, adv_sentences[i], mc_sentences[i])

                    idx += 1

            rewards = torch.tensor(rewards_np, dtype=torch.float)
            if self.gpu:
                rewards = rewards.cuda()

        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        return rewards


    def get_reward_gpt_similarity(self, sentences, windows_size, rate, rollout_num=1):

        adv_sentences = []
        for array_ofwords in tensor_to_tokens(sentences, self.idx2word_dict):
              adv_sentences.append(' '.join(array_ofwords))

        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards_np = np.zeros([self.max_seq_len, batch_size])

            w2v_model = get_w2vmodel('wikidump') #TODO: que sea cfg y se cargue desde la funcion padre

            gpt_sentences = generate_gpt_sentences(adv_sentences, rollout_num * self.max_seq_len, windows_size, rate)
            for num, gpt_sentence_group in enumerate(gpt_sentences): # [0, max_seq_len]
                for i in range(batch_size):
                    cosine_similarity = mc_similarity(w2v_model, adv_sentences[i], gpt_sentence_group[i])

                    if cosine_similarity == 0:
                        rewards_np[num][i] = 0
                    else:
                        cosine_distance = 1 - ((1 - cosine_similarity)/2) # TODO: Realizar normalizacion dentro de `mc_similarity`
                        rewards_np[num][i] = cosine_distance

            rewards = torch.tensor(rewards_np, dtype=torch.float)
            if self.gpu:
                rewards = rewards.cuda()

        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        return rewards


    def get_token_reward(self, sentences, rollout_num, dis, current_k, given_num):
        """
        get reward of each token in sequence via Monte Carlo search
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num, batch_size]).float()
            idx = 0
            for i in range(rollout_num):
                samples = self.rollout_mc_search(sentences, given_num)
                out = dis(samples)
                out = F.softmax(out, dim=-1)
                reward = out[:, current_k + 1]
                rewards[idx] = reward
                idx += 1

        rewards = torch.Tensor(rewards).cuda()
        rewards = torch.sum(rewards, dim=0) / rollout_num
        print("token reward: ", rewards, end=", ")
        return rewards

    def get_reward_csgan(self, target, rollout_num, csgan_clas):
        pass
