from collections import defaultdict
import dynet as dy
import numpy as np
import random
import sys
import datetime

def read_data(filename):
    sentences = []
    with open(filename) as reader:
        for line in reader:
            sentences.append(['<s>'] + line.strip().split() + ['</s>'])
    return sentences

def read_dict(filename):
    mapping = {}
    with open(filename) as reader:
        for line in reader:
            parts = line.strip().split()
            mapping[parts[0]] = parts[1]
    return mapping

class Attention:
    def __init__(self, model, trainer, training_src, training_tgt, emd_size, hid_size, att_size, test_src, blind_src, dict_mapping):
        self.model = model
        self.trainer = trainer
        self.emd_size = emd_size
        self.hid_size = hid_size
        self.att_size = att_size
        self.training = [(x, y) for (x, y) in zip(training_src, training_tgt)]
        self.src_w2i, self.src_i2w = self.get_vocab(training_src)
        self.tgt_w2i, self.tgt_i2w = self.get_vocab(training_tgt)
        self.src_vocab_size = len(self.src_w2i)
        self.tgt_vocab_size = len(self.tgt_w2i)

        self.test_src = test_src
        self.blind_src = blind_src
        self.train_src = training_src
        self.dict_mapping = dict_mapping

        # parameter for lookup table
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.emd_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.emd_size))

        # parameter for encoder and decoder
        self.l2r_builder = dy.LSTMBuilder(1, self.emd_size, self.hid_size, model)
        self.r2l_builder = dy.LSTMBuilder(1, self.emd_size, self.hid_size, model)
        self.dec_builder = dy.LSTMBuilder(1, 2 * self.hid_size + self.emd_size, self.hid_size, model)

        # parameter for output word
        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hid_size))
        self.b_y = model.add_parameters((self.tgt_vocab_size))

        # parameter for attention
        self.W1_att_f = model.add_parameters((self.att_size, 2 * self.hid_size))
        self.W1_att_e = model.add_parameters((self.att_size, self.hid_size))
        self.W2_att = model.add_parameters((1, self.att_size))


    def get_vocab(self, sentences):
        word_freq = self.get_freq(sentences)
        w2i = defaultdict(lambda : 0)
        w2i['<unk>'] = 0
        for word, freq in word_freq.items():
            if freq > 3:
                w2i[word] = len(w2i)
        i2w = {idx : word for (word, idx) in w2i.items()}
        return w2i, i2w


    def get_freq(self, sentences):
        word_freq = {}
        for sent in sentences:
            for word in sent:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        return word_freq

    def attention_mlp(self, h_fs_matrix, h_e, att_enc):
        #W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        W2_att = dy.parameter(self.W2_att)

        att_dec = W1_att_e * h_e
        unnormalized = W2_att * dy.tanh(dy.colwise_add(att_enc, att_dec))
        att_weights = dy.softmax(dy.transpose(unnormalized))
        context = h_fs_matrix * att_weights

        return context

    def attention_mlp_translate(self, h_fs_matrix, h_e, att_enc):
        #W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        W2_att = dy.parameter(self.W2_att)

        att_dec = W1_att_e * h_e
        unnormalized = W2_att * dy.tanh(dy.colwise_add(att_enc, att_dec))
        att_weights = dy.softmax(dy.transpose(unnormalized))
        context = h_fs_matrix * att_weights

        weights = att_weights.vec_value()
        max_idx = weights.index(max(weights))
        return context, max_idx


    def step(self, sent_pair):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_f = dy.parameter(self.W1_att_f)

        src_sent, tgt_sent = sent_pair
        src_sent_rev = list(reversed(src_sent))

        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(src_sent, src_sent_rev):
            #l2r_emb = self.src_lookup[self.src_w2i[cw_l2r]]
            #r2l_emb = self.src_lookup[self.src_w2i[cw_r2l]]
            l2r_emb = dy.lookup(self.src_lookup, self.src_w2i[cw_l2r])
            r2l_emb = dy.lookup(self.src_lookup, self.src_w2i[cw_r2l])
            l2r_state = l2r_state.add_input(l2r_emb)
            r2l_state = r2l_state.add_input(r2l_emb)
            l2r_contexts.append(l2r_state.output())
            r2l_contexts.append(r2l_state.output())

        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)
        att_enc = W1_att_f * h_fs_matrix

        losses = []
        num_words = 0

        c_t = dy.vecInput(2 * self.hid_size)
        #start = dy.concatenate([self.tgt_lookup[self.tgt_w2i['<s>']], c_t])
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_w2i['<s>']), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        for (cw, nw) in zip(tgt_sent, tgt_sent[1:]):
            h_e = dec_state.output()
            c_t = self.attention_mlp(h_fs_matrix, h_e, att_enc)
            #last_emd = self.tgt_lookup[self.tgt_w2i[cw]]
            last_emd = dy.lookup(self.tgt_lookup, self.tgt_w2i[cw])
            x_t = dy.concatenate([last_emd, c_t])
            dec_state = dec_state.add_input(x_t)
            output_probs = dy.softmax(W_y * dec_state.output() + b_y)
            losses.append(-dy.log(dy.pick(output_probs, self.tgt_w2i[nw])))
            num_words += 1

        loss = dy.esum(losses)
        return loss, num_words

    def train(self, num_epoch):
        for i in range(num_epoch):
            print(datetime.datetime.now())
            loss_value = 0
            random.shuffle(self.training)
            for pair in self.training:
                loss, num_words = self.step(pair)
                loss_value += loss.value() / num_words
                loss.backward()
                self.trainer.update()
                #print('epoch %d, loss %f' % (i, loss_value))
            self.translate_test(i)
            print('epoch %d, loss %f' % (i, loss_value / len(self.training)))

    def translate(self, src_sent):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W1_att_f = dy.parameter(self.W1_att_f)

        src_sent_rev = list(reversed(src_sent))

        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(src_sent, src_sent_rev):
            #l2r_emb = self.src_lookup[self.src_w2i[cw_l2r]]
            #r2l_emb = self.src_lookup[self.src_w2i[cw_r2l]]
            l2r_emb = dy.lookup(self.src_lookup, self.src_w2i[cw_l2r])
            r2l_emb = dy.lookup(self.src_lookup, self.src_w2i[cw_r2l])
            l2r_state = l2r_state.add_input(l2r_emb)
            r2l_state = r2l_state.add_input(r2l_emb)
            l2r_contexts.append(l2r_state.output())
            r2l_contexts.append(r2l_state.output())

        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)
        att_enc = W1_att_f * h_fs_matrix

        # decoder
        result = ['<s>']
        cw = result[-1]
        c_t = dy.vecInput(2 * self.hid_size)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_w2i['<s>']), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        while len(result) < 2 * len(src_sent):
            h_e = dec_state.output()
            c_t, max_idx = self.attention_mlp_translate(h_fs_matrix, h_e, att_enc)
            last_emd = dy.lookup(self.tgt_lookup, self.tgt_w2i[cw])
            x_t = dy.concatenate([last_emd, c_t])
            dec_state = dec_state.add_input(x_t)
            output_probs = dy.softmax(W_y * dec_state.output() + b_y).vec_value()
            cw = self.tgt_i2w[output_probs.index(max(output_probs))]
            if cw == '</s>':
                break
            elif cw == '<unk>' and src_sent[max_idx] in self.dict_mapping:
                result.append(self.dict_mapping[src_sent[max_idx]])
            else:
                result.append(cw)

        return ' '.join(result[1:])


    def translate_test(self, epoch_num):
        with open('test/' + str(epoch_num), 'w') as writer:
            for src_sent in self.test_src:
                result = self.translate(src_sent)
                writer.write(result + '\n')
        with open('blind/' + str(epoch_num), 'w') as writer:
            for src_sent in self.blind_src:
                result = self.translate(src_sent)
                writer.write(result + '\n')

def main():
    train_src_file = 'data/en-de/train.en-de.low.de'
    train_tgt_file = 'data/en-de/train.en-de.low.en'
    valid_src_file = 'data/en-de/valid.en-de.low.de'
    valid_tgt_file = 'data/en-de/train.en-de.low.en'
    test_src_file = 'data/en-de/test.en-de.low.de'
    test_tgt_file = 'data/en-de/test.en-de.low.en'
    blind_src_file = 'data/en-de/blind.en-de.low.de'
    dict_file = 'dict_mapping'

    print 'reading data...'
    train_src = read_data(train_src_file)
    train_tgt = read_data(train_tgt_file)
    #valid_src = read_data(valid_src_file)
    #valid_tgt = read_data(valid_tgt_file)
    test_src = read_data(test_src_file)
    #test_tgt = read_data(test_tgt_file)
    blind_src = read_data(blind_src_file)

    dict_mapping = read_dict(dict_file)
    print 'done!'

    EMB_SIZE = 128
    ATT_SIZE = 128
    HID_SIZE = 128
    NUM_EPOCH = 1000
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)
    attention = Attention(model, trainer, train_src, train_tgt, EMB_SIZE, HID_SIZE, ATT_SIZE, test_src, blind_src, dict_mapping)
    attention.train(NUM_EPOCH)

if __name__ == '__main__': main()
