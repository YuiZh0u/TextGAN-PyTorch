# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from models.discriminator import CNNDiscriminator


class SeqGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, gpu=False, dropout=0.2):
        super(SeqGAN_D, self).__init__(embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, gpu, dropout)