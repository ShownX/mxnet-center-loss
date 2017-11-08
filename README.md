# MxNet implementation of the paper: A Discriminative Feature Learning Approach for Deep Face Recognition

## Requirements
```
pip install -r requirements.txt
```

## Training
1. Train with original softmax
```
$ python main.py --train --prefix=softmax
```

2. Train with softmax + center loss
```
$ python main.py --train --center_loss --prefix=center-loss
```

## Test
1. Test with original softmax
```
$ python main.py --test --prefix=softmax
```

2. Test with softmax + center loss
```
$ python main.py --test --prefix=center-loss
```

## Image
Comparison Accuracy curve:

<img src="output/curves.png"></img>

### Softmax
Training:

<img src="output/softmax-train.gif"></img>

Testing:

<img src="output/softmax-test.gif"></img>

### Softmax + Center Loss
Training:

<img src="output/center-train.gif"></img>

Testing:

<img src="output/center-test.gif"></img>

