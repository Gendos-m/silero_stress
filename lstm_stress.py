import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime

def create_output(y_pred, ind_col):
    out_df = pd.DataFrame({'stress': y_pred})
    out_df.index = ind_col
    return out_df

def create_model(x_vec_shape):
    model = Sequential()
    # model.add(Reshape((x_vec_shape[0] * x_vec_shape[1], 1), input_shape=(x_vec_shape)))
    # model.add(Embedding(input_dim=32, output_dim=256, input_length=(x_vec_shape[0] * x_vec_shape[1])))  # x_vec_shape[1] = inp_words,
    # model.add(Reshape())
    # model.add(LSTM(64, activation='ReLU', dropout=0.2, input_shape=(x_vec_shape)))
    model.add(LSTM(128, activation='ReLU', input_shape=(x_vec_shape)))
    model.add(Dense(6, activation='softmax'))
    model.summary()

    return model

def evaluate(model, csv_input, strs_letters, rst_letters, max_word, max_syll_num):
    words_ = csv_input['word'].values
    lemmas_ = csv_input['lemma'].values
    num_syllables = csv_input['num_syllables'].values
    test_ftr_vec = create_features(words_, lemmas_, strs_letters,
                    rst_letters, max_word)
    single_indices = np.where(num_syllables == 1)[0]  # индексы слов с одним слогом
    single_pattern = np.zeros(max_syll_num)  # паттерн для односложных слов
    single_pattern[0] = 1
    preds = model.predict(test_ftr_vec, verbose=1)
    preds[single_indices, :] = single_pattern  # замена паттернов с одним слогом верным паттерном
    stress_position = np.argmax(preds, axis=1) + 1
    return stress_position



def one_point_encoding(input_word, max_len_wrd, basis_features):  # по сути, кодирование слова по позиции буквы и самой букве
    feature_init = np.zeros((max_len_wrd, len(basis_features))).astype(int)

    for i, ltr in enumerate(input_word):
        ind = basis_features.find(ltr)
        feature_init[i, ind] = 1

    return feature_init

def calc_max_word_len(csv_file):  # максимальная длина слова из всей выборки
    wrd_max_len_train = np.max([len(i) for i in csv_file['word']]) + 1
    return wrd_max_len_train

def create_labels(strs_let, max_syll_num):  #  генерация y вектора
    lbls_np = np.zeros((len(strs_let), max_syll_num)).astype(int)

    for i, let in enumerate(strs_let):
        lbls_np[i, let-1] = 1

    return lbls_np


def create_features(all_wrds, all_lemmas, strs_let, rst_let, max_word_len):  # функция генерит вектор признаков для конкретного слова
    # sep_trn_words = []  # это список слов по "слогам"
    basis_let = strs_let + rst_let
    ftr_vec = []
    for (wrd, lemma) in zip(all_wrds, all_lemmas):  # делим слово на "слоги" простым методом - добавлением согласных с 2-х сторон

        wrd_len = len(wrd)
        vowel_inds = [i for i, ltr in enumerate(wrd) if ltr in strs_let]  # позиции гласных в слове
        wrd_parts = []  # список "слогов" в слове
        for j, v_s in enumerate(vowel_inds):
            lft_bnd = v_s - 1 if v_s - 1 >= 0 and v_s - 1 not in vowel_inds else v_s  # левая граница слога
            rght_bnd = v_s + 1 if v_s + 1 <= wrd_len and v_s + 1 not in vowel_inds else v_s  # правая граница слога
            wrd_part = wrd[lft_bnd: rght_bnd + 1]  # выделение слога
            wrd_parts.append(wrd_part)

        word_encoding = one_point_encoding(wrd, max_word_len, basis_let)
        lemma_encoding = one_point_encoding(lemma, max_word_len, basis_let)
        word_accent = create_accent_feature(wrd, wrd_parts, max_word_len)

        # feature_vec = np.concatenate([word_encoding, word_accent[:, np.newaxis]], axis=1)
        feature_vec = np.concatenate([word_encoding, lemma_encoding, word_accent[:, np.newaxis]], axis=1)
        # feature_vec = np.concatenate([word_encoding, lemma_encoding], axis=1)

        ftr_vec.append(feature_vec)

    ftr_vec_np = np.array(ftr_vec)
    return ftr_vec_np  # (количество слов Х количество признаков Х максимальная длина слова)


def create_accent_feature(input_word, input_word_in_parts, wrd_len_max):
    part_inicators = np.zeros(wrd_len_max).astype(int)
    tmp_wrd = input_word
    shift = 0
    for i, prt in enumerate(input_word_in_parts):
        prt_len = len(prt)
        if shift - 1 > 0:  # случай, если обрезана нужная буква первым слогом
            ind_ = tmp_wrd[shift - 1:].find(prt)
        else:
            ind_ = tmp_wrd[shift:].find(prt)
        part_inicators[shift + ind_: shift + ind_ + prt_len] = i + 1
        shift += ind_ + prt_len

        for_test = part_inicators[:len(input_word)]
        nulls_inds = np.where(for_test == 0)[0]  # проверка, есть ли мягкий или твердый знак не вошедший ни в один слог
        for ind in nulls_inds:  # заменяем ноль на ближайший номер слога слева
            if ind == 0:  # самая первая буква
                part_inicators[ind] = 1

            else:  # не первая буква
                part_inicators[ind] = part_inicators[ind - 1]

    return part_inicators

max_num_syllables = 6
path_root = '/Users/mamykingennady/Desktop/ПОДГОТОВКА К СОБЕСАМ/Speech_syntex'
test_csv = pd.read_csv(path_root + '/test.csv',
                       header=0, index_col=0)
train_csv = pd.read_csv(path_root + '/train.csv',
                        header=0, index_col=0)

# ______создание тренировочных данных___
one_syll_part = train_csv[train_csv['num_syllables'] != 1]  # убираем все слова с одним слогом - очевидно ударение
# correct_wrds = [wrd.replace('ё', 'e') for wrd in one_syll_part['word'].values if always_stress in wrd]

# _______создание базиса features
all_trn_words = one_syll_part['word'].values  # извлекаем массив слов
all_trn_stresses = one_syll_part['stress'].values  # извлекаем массив ударений
all_trn_sylls = one_syll_part['num_syllables'].values  # извлекаем массив количества слогов
all_trn_lemmas = one_syll_part['lemma'].values  # извлекаем массив лемм

stress_letters = 'уеыаоэёяию'  # по сути гласные
rest_letters = 'йцкнгшщзхъфвпрлджчсмтьб'  # все остальные
max_wrd_len = max(calc_max_word_len(train_csv), calc_max_word_len(test_csv))

features_vec = create_features(all_trn_words, all_trn_lemmas, stress_letters,
                               rest_letters, max_wrd_len)  # разделение по слогам

X_vec = features_vec
Y_vec = create_labels(all_trn_stresses, max_num_syllables)

X_tr, X_val, y_tr, y_val = train_test_split(X_vec, Y_vec, test_size=0.2)

model = create_model(X_vec.shape[-2:])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Change the number of epochs and the batch size depending on the RAM Size
d1 = datetime.now()
batch_size = 32
history = model.fit(X_tr, y_tr, epochs=8, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val))
pred_arr = evaluate(model, test_csv, stress_letters, rest_letters, max_wrd_len, max_num_syllables)
output_df = create_output(pred_arr, test_csv.index)
d2 = datetime.now()
print(d2-d1)

output_df.to_csv('output.csv')