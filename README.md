# A framework for all papers
## DefensiveRAW: Robust Image Authentication via Transferable RAW Protection

### Mode Define
- mode=0: generating protected images (val)
- mode=1: tampering localization on generating protected images (val)
- mode=2: regular training, including ISP, RAW2RAW and localization (train)
- mode=3: regular training for ablation (RGB protection), including RAW2RAW and localization (train)
- mode=4: OSN performance (val)
- mode=5: train a ISP using restormer for validation (train)

### Issues
- restormer cannot be loaded simultaneously with OSN network, because they share the same variable ```localizer```

### notes

How to test OSN in the code
- run bash ./run_ISP_OSN.sh (mode==4)
- Line 141 of Modified_invISP.py, modify the model as that of OSN network
- specify Line 1319-2323 which provides the tamper source and mask
- the setting file is train_ISP_OSN.yml. If you want to do automatic copy-move, set ```inference_tamper_index=2``` and ```inference_load_real_world_tamper=False```
- ```using_which_model_for_test``` decides using which model for testing. ```discriminator_mask``` is our method, ```localizer``` is OSN.
- The average F1 score will be printed in the console
- The main loop loops over the training set. Therefore you should manually kill the process when all the validate images are runned.

- the flow is optimize_parameters_router -> get_performance_of_OSN
- use gpu 0 as default

Testing the baseline (RGB protection)

### Logs
#### 1003

- test is now ok. Use ```mode=0```, which would contain protected image generation and tampering localization.

#### 1012
- update option and introduce base option

