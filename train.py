# coding=utf-8
from __future__ import print_function
import optparse
import itertools
from collections import OrderedDict
import loader
import torch
import time
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
# import visdom
from utils import *
from loader import *
from model import BiLSTM_CRF
# import wandb to log experiments
import wandb
import datetime

t = time.time()
models_path = "models/"

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="dataset/train.txt",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="dataset/dev.txt",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="dataset/test.txt",
    help="Test set location"
)
# optparser.add_option(
#     '--test_train', default='dataset/eng.train54019',
#     help='test train'
# )
optparser.add_option(
    '--score', default='evaluation/temp/score.txt',
    help='score file location'
)
optparser.add_option(
    "-s", "--tag_scheme", default="iob",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="200",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="models/glove.6B.100d.txt",
    help="Location of pretrained embeddings"
)

# optparser.add_option(
#     '--pre_emb', type=str, default='random', 
#     help='use pretrained char embedding or init it randomly'
#     )


optparser.add_option(
    "-A", "--all_emb", default="1",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--loss', default='loss.txt',
    help='loss file location'
)
optparser.add_option(
    '--name', default='test',
    help='model name'
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
opts = optparser.parse_args()[0]

parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 0
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['reload'] = opts.reload == 1
parameters['name'] = opts.name
parameters['char_mode'] = opts.char_mode

parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
use_gpu = parameters['use_gpu']

mapping_file = 'models/mapping.pkl'

name = parameters['name']
model_name = models_path + name #get_name(parameters)
tmp_model = model_name + '.tmp'

# Change some parameters only once, to include in WandB as well
param_num_epochs = 10
param_learning_rate = 0.015
param_plot_every = 1000
param_eval_every = 2000
param_momentum = 0.9

# log training to wandb
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
run_name = f"{parameters['name']}-{timestamp}"
wandb.init(project='RTB-NER-Transfer-Learning', name=run_name)


assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
# assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)
# test_train_sentences = loader.load_sentences(opts.test_train, lower, zeros)

update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)
# update_tag_scheme(test_train_sentences, tag_scheme)

# if parameters['pre_emb']:
#     dico_words_train = word_mapping(train_sentences, lower)[0]
#     dico_words, word_to_id, id_to_word = augment_with_pretrained(
#         dico_words_train.copy(),
#         parameters['pre_emb'],
#         list(itertools.chain.from_iterable(
#             [[w[0] for w in s] for s in dev_sentences + test_sentences])
#         ) if not parameters['all_emb'] else None
#     )

# else:
dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
dico_words_train = dico_words

# dico_words_train = word_mapping(train_sentences, lower)[0]

# dico_words, word_to_id, id_to_word = augment_with_pretrained(
#         dico_words_train.copy(),
#         parameters['pre_emb'],
#         list(itertools.chain.from_iterable(
#             [[w[0] for w in s] for s in dev_sentences + test_sentences])
#         ) if not parameters['all_emb'] else None
#     )

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)
# test_train_data = prepare_dataset(
#     test_train_sentences, word_to_id, char_to_id, tag_to_id, lower
# )

print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

# if parameters['pre_emb']:
#     all_word_embeds = {}
#     for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
#         s = line.strip().split()
#         if len(s) == parameters['word_dim'] + 1:
#             all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

word_embeds = random_embedding(word_to_id, opts.word_dim)
# word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), opts.word_dim))



# for w in word_to_id:
#     if w in all_word_embeds:
#         word_embeds[word_to_id[w]] = all_word_embeds[w]
#     elif w.lower() in all_word_embeds:
#         word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

# print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    pickle.dump(mappings, f)

print('word_to_id: ', len(word_to_id))

# Log parameters
wandb.config.epochs = param_num_epochs
wandb.config.vocab_size = len(word_to_id)
wandb.config.embedding_dim = parameters['word_dim']
wandb.config.hidden_dim = parameters['word_lstm_dim']
wandb.config.use_crf = parameters['crf']
wandb.config.char_mode = parameters['char_mode']
wandb.config.learning_rate = param_learning_rate
wandb.config.use_gpu = use_gpu
wandb.config.plot_every = param_plot_every
wandb.config.eval_every = param_eval_every
wandb.config.momentum = param_momentum
wandb.config.optimizer = "SGD"


model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'],
                   char_mode=parameters['char_mode'])
                   # n_cap=4,
                   # cap_embedding_dim=10)
if parameters['reload']:
    model.load_state_dict(torch.load(model_name))

# Log gradients and model parameters
wandb.watch(model)

if use_gpu:
    model.cuda()
learning_rate = param_learning_rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=param_momentum)
losses = []
loss = 0.0
best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0
all_F = [[0, 0, 0]]
plot_every = param_plot_every
eval_every = param_eval_every
count = 0
# vis = visdom.Visdom()
sys.stdout.flush()


def evaluating(model, datas, best_F):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))

    # Non-O entities evaluation
    total_entities = 0
    total_non_O_entities = 0
    correct_entities = 0
    correct_non_O_entities = 0

    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)

        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1

            # Non-O entities evaluation
            total_entities += 1
            if id_to_tag[true_id] != 'O':
                total_non_O_entities += 1

            if true_id == pred_id:
                correct_entities += 1
                if id_to_tag[true_id] != 'O':
                    correct_non_O_entities += 1

        prediction.append('')
    predf = eval_temp + '/pred.' + name
    scoref = eval_temp + '/score.' + name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))

    # Non-O entities evaluation
    overall_accuracy = correct_entities / total_entities * 100.0
    non_O_accuracy = correct_non_O_entities / total_non_O_entities * 100.0

    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Non-'O' entities accuracy: {non_O_accuracy:.2f}%")

    return best_F, new_F, save


model.train(True)

for epoch in range(1, param_num_epochs + 1):
    print('='*80)
    print("epoch = %i." % epoch)
    print('='*80)
    for i, index in enumerate(np.random.permutation(len(train_data))):
        tr = time.time()
        count += 1
        data = train_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']
        chars2 = data['chars']

        # ######## char lstm
        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        # ######## char cnn
        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))


        targets = torch.LongTensor(tags)
        caps = Variable(torch.LongTensor(data['caps']))
        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d)
        else:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d)
        # loss += neg_log_likelihood.data[0] / len(data['words'])
        loss += neg_log_likelihood.data / len(data['words'])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if count % plot_every == 0:
            loss /= plot_every
            print(count, ': ', loss)

            # Log the averaged loss and current count to wandb
            wandb.log({"avg_loss": loss, "iteration": count})

            if losses == []:
                losses.append(loss)
            losses.append(loss)
            text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
            losswin = 'loss_' + name
            textwin = 'loss_text_' + name
            # vis.line(np.array(losses), X=np.array([plot_every*i for i in range(len(losses))]),
                #  win=losswin, opts={'title': losswin, 'legend': ['loss']})
            # vis.text(text, win=textwin, opts={'title': textwin})
            loss = 0.0

        if count % (eval_every) == 0 and count > (eval_every * 20) or \
                count % (eval_every*4) == 0 and count < (eval_every * 20):
            model.train(False)
            # best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
            best_train_F, new_train_F, _ = evaluating(model, train_data, best_train_F)
            best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F)
            if save:
                torch.save(model, model_name)
            best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)

            # Log the new F-scores to wandb
            wandb.log({"train_F": new_train_F, "dev_F": new_dev_F, "test_F": new_test_F, "iteration": count})

            sys.stdout.flush()

            # all_F.append([new_train_F, new_dev_F, new_test_F])
            # Fwin = 'F-score of {train, dev, test}_' + name
            # vis.line(np.array(all_F), win=Fwin,
                #  X=np.array([eval_every*i for i in range(len(all_F))]),
                #  opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
            model.train(True)

        if count % len(train_data) == 0:
            adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))

    # log the epochs
    wandb.log({"epoch": epoch})

print(time.time() - t)

# plt.plot(losses)
# plt.show()

# Close Weights and Biases
wandb.finish()
