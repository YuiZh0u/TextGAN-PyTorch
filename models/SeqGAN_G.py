# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn.functional as F

from models.generator import LSTMGenerator


class SeqGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SeqGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'seqgan'

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """
        print("SeqGAN_G inp: ", inp)
        print("SeqGAN_G inp size: ", inp.size())
        print("SeqGAN_G target: ", target)
        print("SeqGAN_G target size: ", target.size())
        print("SeqGAN_G reward: ", reward)
        print("SeqGAN_G reward size: ", reward.size())
        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)
        print("SeqGAN_G batch_size: ", batch_size)
        print("SeqGAN_G seq_len: ", seq_len)
        print("SeqGAN_G hidden: ", hidden)

        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        print("SeqGAN_G out: ", out)
        print("SeqGAN_G out size: ", out.size())
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        print("SeqGAN_G target_onehot: ", target_onehot)
        print("SeqGAN_G target_onehot size: ", target_onehot.size())
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        print("SeqGAN_G pred: ", pred)
        print("SeqGAN_G pred size: ", pred.size())
        loss = -torch.sum(pred * reward)
        print("SeqGAN_G loss: ", loss)
        print("SeqGAN_G loss size: ", loss.size())

        return loss
