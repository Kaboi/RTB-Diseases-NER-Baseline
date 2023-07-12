import spacy
import random
import wandb
from spacy.training.example import Example
from spacy.training import Corpus
from spacy import displacy
from spacy.util import minibatch
from spacy.util import prefer_gpu
from pathlib import Path
import tempfile
import shutil
import itertools
import matplotlib.pyplot as plt
import numpy as np

# Use GPU if available
if prefer_gpu():
    print("Using GPU!")
else:
    print("Using CPU :(")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
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
    return fig

# Initialize wandb
wandb.init(project="spacy-ner-training")

# Load a blank English model
nlp = spacy.blank("en")

# Create a new NER pipe and add it to the pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)

# Load the training and evaluation data (assuming they are in SpaCy binary format)
train_corpus = Corpus("/path/to/your/training_data.spacy")
eval_corpus = Corpus("/path/to/your/evaluation_data.spacy")

# Add the labels to the NER pipe
for label in train_corpus.get_labels(nlp):
    ner.add_label(label)

# Train the model
optimizer = nlp.begin_training()

best_f1 = 0.0
best_epoch = 0

for epoch in range(25):
    losses = {}
    random.shuffle(train_corpus)
    for batch in minibatch(train_corpus(nlp), size=8):
        for example in batch:
            nlp.update([example], losses=losses, drop=0.5)
    # Evaluate the model
    eval_scores = nlp.evaluate(eval_corpus(nlp))
    wandb.log({"epoch": epoch, "loss": losses, "eval_scores": eval_scores.scores})
    # If the F1 score of the current epoch is better than the best F1 score, save the model
    if eval_scores["ents_f"] > best_f1:
        best_f1 = eval_scores["ents_f"]
        best_epoch = epoch
        # Log the best model to wandb as an artifact
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            nlp.to_disk(temp_path)
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_dir(temp_path)
            wandb.run.log_artifact(artifact)

# Reload the best model
with tempfile.TemporaryDirectory() as tempdir:
    temp_path = Path(tempdir)
    best_model_artifact = wandb.run.use_artifact('best_model:type=model')
    best_model_artifact.download(target_dir=temp_path)
    nlp = spacy.load(temp_path)

# Evaluate the best model on the evaluation dataset
best_eval_scores = nlp.evaluate(eval_corpus(nlp))
wandb.log({"best_eval_scores": best_eval_scores.scores})

# Create confusion matrix using the provided function and log to wandb
confusion_matrix = np.array(best_eval_scores["ents_per_type"])
fig = plot_confusion_matrix(confusion_matrix, classes=ner.labels, normalize=True)
wandb.log({"final_confusion_matrix": wandb.Image(fig)})

# Optionally, you can also display the confusion matrix
plt.show()

# End the W&B run
wandb.finish()
