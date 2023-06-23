# coding=utf-8
from __future__ import print_function
import optparse
import torch
import time
import pickle
from torch.autograd import Variable
from loader import *
from utils import *
# import wandb to log experiments
import wandb
import datetime
# import for hq creating confusion matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np


t = time.time()

# python -m visdom.server

optparser = optparse.OptionParser()
optparser.add_option(
    "-t", "--test", default="dataset/test.txt",
    help="Test set location"
)
optparser.add_option(
    '--score', default='evaluation/temp/score.txt',
    help='score file location'
)
optparser.add_option(
    "-f", "--crf", default="0",
    type='int', help="Use CRF (0 to disable)"
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
    '--model_path', default='models/lstm_crf.model',
    help='model path'
)
optparser.add_option(
    '--map_path', default='models/mapping.pkl',
    help='model path'
)
optparser.add_option(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)

opts = optparser.parse_args()[0]

mapping_file = opts.map_path

with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

use_gpu = opts.use_gpu == 1 and torch.cuda.is_available()


assert os.path.isfile(opts.test)
assert parameters['tag_scheme'] in ['iob', 'iobes']

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

test_sentences = load_sentences(opts.test, lower, zeros)
update_tag_scheme(test_sentences, tag_scheme)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

# log evaluation to wandb
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
run_name = f"{parameters['name']}-{timestamp}-eval"
# wandb.init(project='RTB-NER-Transfer-Learning', name=run_name)
wandb.init(mode="disabled")

# Log parameters
wandb.config.char_mode = parameters['char_mode']
wandb.config.use_gpu = use_gpu


model = torch.load(opts.model_path)
model_name = opts.model_path.split('/')[-1].split('.')[0]

if use_gpu:
    model.cuda()
model.eval()

# def getmaxlen(tags):
#     l = 1
#     maxl = 1
#     for tag in tags:
#         tag = id_to_tag[tag]
#         if 'I-' in tag:
#             l += 1
#         elif 'B-' in tag:
#             l = 1
#         elif 'E-' in tag:
#             l += 1
#             maxl = max(maxl, l)
#             l = 1
#         elif 'O' in tag:
#             l = 1
#     return maxl


def get_entities(seq):
    """Gets entities from sequence."""
    entities = []
    entity = []
    for i, tag in enumerate(seq):
        if tag.startswith('B-'):
            if entity:
                entities.append(entity)
            entity = [tag[2:], i]
        elif tag.startswith('I-'):
            if entity and entity[0] == tag[2:]:
                entity.append(i)
            else:
                if entity:
                    entities.append(entity)
                entity = [tag[2:], i]
        else:
            if entity:
                entities.append(entity)
            entity = []
    if entity:
        entities.append(entity)
    return entities


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# def get_entity_mapping(entities):
#     entity_to_id = {k: v for v, k in enumerate(set(entities))}
#     id_to_entity = {v: k for k, v in entity_to_id.items()}
#     entity_labels = list(entity_to_id.keys())
#     return entity_to_id, id_to_entity, entity_labels

def get_entity_mapping(entities):
    entities = [tuple(e) if isinstance(e, list) else e for e in entities]
    unique_entities = dict.fromkeys(entities).keys()  # maintains the order and removes duplicates
    entity_to_id = {k: v for v, k in enumerate(unique_entities)}
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    entity_labels = list(unique_entities)
    return entity_to_id, id_to_entity, entity_labels


def compute_entity_confusion_matrix(ground_truth_entity_ids, predicted_entity_ids, labels):
    """Compute confusion matrix for entities."""
    assert len(ground_truth_entity_ids) == len(predicted_entity_ids), "Length of ground truth IDs and predicted IDs " \
                                                                      "must be the same."
    assert max(labels) + 1 == len(labels), "Labels should be a list of unique integers from 0 to num_classes - 1."

    # Determine the number of unique entity classes from the labels.
    num_classes = len(labels)

    # Initialize confusion matrix with zeros.
    confusion_matrix = torch.zeros((num_classes, num_classes))

    # Iterate over all ground truth IDs and predicted IDs and increment corresponding cell in confusion matrix.
    for gt, pred in zip(ground_truth_entity_ids, predicted_entity_ids):
        confusion_matrix[gt, pred] += 1

    return confusion_matrix


# def eval(model, datas, maxl=1):
def eval(model, datas):
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))

    # Collecting all ground truth ids and predicted ids for all datas
    ground_truth_ids = []
    predicted_ids = []

    # Collecting all ground truth and predicted entities for all datas
    ground_truth_entities = []
    predicted_entities = []

    for data in datas:
        ground_truth_id = data['tags']
        # l = getmaxlen(ground_truth_id)
        # if not l == maxl:
        #     continue
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
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(),chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out

        # Append current batch of ground truth and predictions to respective lists
        ground_truth_ids.extend(ground_truth_id)
        predicted_ids.extend(predicted_id)

        # Compute entities for ground truth and prediction
        ground_truth_entities.extend(get_entities([id_to_tag[gti] for gti in ground_truth_id]))
        predicted_entities.extend(get_entities([id_to_tag[pi] for pi in predicted_id]))

        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')

    # Log the confusion matrix to Weights & Biases
    wandb.log({"individual_level_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=ground_truth_ids,
        preds=predicted_ids,
        class_names=[id_to_tag[i] for i in range(len(tag_to_id) - 2)]
    )})

    # Assuming you have a way to convert entities to indices and the entity_labels list
    entity_to_id, id_to_entity, entity_labels = get_entity_mapping(ground_truth_entities + predicted_entities)

    ground_truth_entities = [tuple(entity) if isinstance(entity, list) else entity for entity in ground_truth_entities]
    predicted_entities = [tuple(entity) if isinstance(entity, list) else entity for entity in predicted_entities]

    missing_entities = set()

    try:
        ground_truth_entity_ids = [entity_to_id[e] for e in ground_truth_entities]
        predicted_entity_ids = [entity_to_id[e] for e in predicted_entities]
    except KeyError as e:
        missing_entities.add(e.args[0])

    print("Missing entities:", missing_entities)
    print(len(ground_truth_ids), len(predicted_ids))
    print(len(ground_truth_entities), len(predicted_entities))
    print(len(ground_truth_entity_ids), len(predicted_entity_ids))

    wandb.log({"entity_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=ground_truth_entity_ids,
        preds=predicted_entity_ids,
        class_names=entity_labels
    )})

    predf = eval_temp + '/pred.' + model_name
    scoref = eval_temp + '/score.' + model_name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    with open(scoref, 'r') as f:
        for l in f.readlines():
            print(l.strip())

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
    print("\n")

    cm = confusion_matrix.numpy()  # Assuming your confusion_matrix is a PyTorch tensor
    plot_confusion_matrix(cm, classes=[id_to_tag[i] for i in range(len(tag_to_id) - 2)])
    plt.savefig("individual_label_confusion_matrix.png", dpi=300)
    wandb.log({"individual_label_confusion_matrix": wandb.Image("individual_label_confusion_matrix.png")})

    cm_entity = compute_entity_confusion_matrix(ground_truth_entity_ids, predicted_entity_ids,
                                                list(range(len(entity_labels))))

    plot_confusion_matrix(cm_entity, classes=entity_labels)
    plt.savefig("entity_confusion_matrix.png", dpi=300)
    wandb.log({"entity_confusion_matrix": wandb.Image("entity_confusion_matrix.png")})


# for l in range(1, 6):
#     print('maxl=', l)
#     eval(model, test_data, l)
#     # print()
# for i in range(10):
#     eval(model, test_data, 100)

# eval(model, test_data, 100)
eval(model, test_data)

# Close Weights and Biases
wandb.finish()

print(time.time() - t)

