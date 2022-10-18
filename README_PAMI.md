
## TPAMI: IMUGE

### Mode Define
- mode=0: training
- mode=1: validation
- mode=2: training KD-JPEG

### Issues

- todo

### How-tos

- todo

### Logs

- (1018: COPY-SPLICING) added ```copysplicing```
- (1017: ERROR SUPERVISION) added ```StopIterationError``` when loss is larger than a threshold.
- (1018: EXP WEIGHT) weight for backward_loss is now defined by ```exponential_weight_for_backward```
- (1017: EASE ATTACKS) remove attacks less than 28dB to alleviate exploding gradient

