# BLSTM-CRF-NER
## reference:

Original paper:

Neural Architectures for Named Entity Recognition<br/>
End-toEnd Sequence labeling via BLSTM-CNN-CRF<br/>
   
code:
Source code is a modified version of what is available at
https://github.com/smutuvi/NER_Baseline<br/>

## requirement

python3.6

pytorch

## Data Format

train data at `./dataset/train.txt`

```
<word> <label>
```


## usage:

train model:

```
python train.py --char_mode 'LSTM' --crf '1'
```

Evaluate model
```
python eval.py --char_mode 'LSTM'
```
query:


1. run the server
	
	```
	python server.py
	```
	
2. GET Method:
	
	```
	http://166.111.5.228:5011/query/<query>
	```
	
3. Return in json, Example:

	```
	http://166.111.5.228:5011/query/search some selection Thomas Edison State College 1902 Goel Shom 's papers
	{"LOC": ["selection", "1902"], "PER": ["Goel"], "CON": ["Shom's"], "DATE": [], "ORG": ["Edison", "State"], "KEY": ["Thomas", "College"], "O": ["search", "some", "papers"]}
	```
	
## File orgnization

```
|- train.py 
|- debug.py 
|- [dir] dataset (word library)
|- [dir] evaluation (help tools when training)
|- [dir] models (well-trained models)
```
