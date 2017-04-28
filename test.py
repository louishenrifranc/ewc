import tensorflow as tf
import model
import pickle
import os
import main
import numpy as np

model_to_restore = os.path.join("model", "model_exp1-0")


def encrypt_single(string, symbol_to_idx):
    return np.array([symbol_to_idx.get(char, 1) for char in string.lower()])


def find_str(s, char):
    index = 0
    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index + len(char)] == char:
                    return index
            index += 1
    return -1


def decrypt_single(sentence, idx_to_symbol):
    return "".join([idx_to_symbol[idx] for idx in sentence])


if __name__ == '__main__':
    with open(os.path.join('Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    with open(os.path.join('Data', 'MovieQA', 'chars_to_idx.pkl'), 'rb') as f:
        char_to_idx = pickle.load(f)

    buckets = [(100, 100)]

    cfg = main.cfg

    model = model.Seq2Seq(buckets, cfg)
    model.build()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_to_restore)

    while True:
        sentence = input("New question: ")
        len_sentence = len(sentence)
        if len_sentence > buckets[0][0]:
            sentence = sentence[:buckets[0][0]]

        q = encrypt_single(sentence, char_to_idx)
        print(q)
        a = encrypt_single("", char_to_idx)

        encoder_size, decoder_size = buckets[0]
        q_pads = np.zeros([1, encoder_size])
        a_pads = np.zeros([1, decoder_size])
        q_pads[0][:q.shape[0]] = q
        a_pads[0][:a.shape[0]] = a
        print(q_pads)

        res = model.forward_with_feed_dict(sess, q_pads, a_pads, is_training=False, ewc_loss_coeff=0)

        outputs = res["predictions"]
        outputs = np.squeeze(outputs)
        outputs = np.argmax(outputs, axis=1)

        output_string = decrypt_single(list(outputs), idx_to_char)

        end_index = find_str(output_string, '<EOS>')
        print("Before removing: ", output_string)
        if end_index == -1:
            print(output_string)
        else:
            print(output_string[:end_index])
