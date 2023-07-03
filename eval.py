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
wandb.init(project='RTB-NER-Transfer-Learning', name=run_name, tags=['baseline', 'eval'])
# wandb.init(project='RTB-NER-Transfer-Learning', name=run_name, mode='disabled')

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

    fig, ax = plt.subplots(figsize=(14, 12))  # Adjust the figsize parameter

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)  # Use the mappable object 'im' for colorbar creation
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=60)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return fig  # Return the figure object





# def eval(model, datas, maxl=1):
def evaluate(model, datas):
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))

    # Collecting all ground truth ids and predicted ids for all datas for logging
    ground_truth_ids = []
    predicted_ids = []

    # Non-O entities evaluation
    total_entities = 0
    total_non_O_entities = 0
    correct_entities = 0
    correct_non_O_entities = 0

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
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)

        predicted_id = out
        # Append current batch of ground truth and predictions to respective lists
        ground_truth_ids.extend(ground_truth_id)
        predicted_ids.extend(predicted_id)

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

    # Log the confusion matrix to Weights & Biases
    wandb.log({"individual_level_confusion_matrix": wandb.plot.confusion_matrix(
        y_true=ground_truth_ids,
        preds=predicted_ids,
        class_names=[id_to_tag[i] for i in range(len(tag_to_id) - 2)]
    )})

    predf = eval_temp + '/pred.' + model_name
    scoref = eval_temp + '/score.' + model_name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    # with open(scoref, 'r') as f:
    #     for l in f.readlines():
    #         print(l.strip())
    #

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    # Logging to wandb
    # Define columns for wandb Table
    columns = ["Entity", "Precision", "Recall", "F1", "Count"]
    # Create an empty list to store rows of the table
    table_rows = []
    # # Placeholder for best_F
    # best_F_wandb = None

    for i, line in enumerate(eval_lines):
        print(line)

        # Logging to wandb
        if i == 1:
            metrics = line.split(';')
            accuracy = float(metrics[0].split(':')[1].strip().replace("%", ""))
            precision = float(metrics[1].split(':')[1].strip().replace("%", ""))
            recall = float(metrics[2].split(':')[1].strip().replace("%", ""))
            fb1 = float(metrics[3].split(':')[1].strip())
            # best_F_wandb = fb1
            # Log accuracy, precision, recall, and fb1 to wandb
            wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "FB1": fb1})
        elif i > 1 and line.strip():  # Skip the first line and empty lines
            entity_metrics = line.split()
            entity_name = entity_metrics[0].replace(":", "")
            precision = float(entity_metrics[2].replace(";", "").replace("%", ""))
            recall = float(entity_metrics[4].replace(";", "").replace("%", ""))
            f1 = float(entity_metrics[6])
            count = int(entity_metrics[7])

            # Append the row to table_rows
            table_rows.append([entity_name, precision, recall, f1, count])

    # Log the table to wandb
    # Create a wandb Table
    metrics_table = wandb.Table(columns=columns, data=table_rows)

    # Log the table to wandb
    wandb.log({"Entity Metrics": metrics_table})


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

    # Non-O entities evaluation
    overall_accuracy = correct_entities / total_entities * 100.0
    non_O_accuracy = correct_non_O_entities / total_non_O_entities * 100.0

    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Non-'O' entities accuracy: {non_O_accuracy:.2f}%")
    wandb.log("non_O_accuracy", non_O_accuracy)

    cm = confusion_matrix.numpy()  # Assuming your confusion_matrix is a PyTorch tensor
    fig = plot_confusion_matrix(cm, normalize=True, classes=[id_to_tag[i] for i in range(len(tag_to_id) - 2)])
    wandb.log({"individual_label_confusion_matrix": wandb.Image(fig)})


# for l in range(1, 6):
#     print('maxl=', l)
#     eval(model, test_data, l)
#     # print()
# for i in range(10):
#     eval(model, test_data, 100)

# eval(model, test_data, 100)
evaluate(model, test_data)

# Close Weights and Biases
wandb.finish()

print(time.time() - t)
