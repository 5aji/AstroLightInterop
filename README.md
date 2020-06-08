# AstroLightInterop

Tools to operate on datasets between light classification models.

Consists of `data converters` and `model interfaces`

### Data converters

These modify datasets to fit a different format for other models.



| Input      | RAPID format | PLAsTiCC format |
|------------|--------------|-----------------|
| PLAsTiCC   | yes          | ---             |
| SPCC       | via PLAsTiCC | planned         |
| SuperNNova | via PLAsTiCC | planned         |
|            |              |                 |
#### Usage
TODO

### Model interfaces

These serve as a standard way to evaluate different models. 

| Model | Testing | Training
| ----- | ------- | -------- |
| RAPID | yes | in progress |
| AVOCADO | planned | planned |
| SuperNNova | planned | planned |

#### Usage
All models take PLAsTiCC-formatted data as their input.


