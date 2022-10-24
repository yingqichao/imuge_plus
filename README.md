# A framework for all papers
## DefensiveRAW: Robust Image Authentication via Transferable RAW Protection

### Mode Define
- mode=0: generating protected images (val)
- mode=1: tampering localization on generating protected images (val)
- mode=2: regular training, including ISP, RAW2RAW and localization (train)
- mode=3: regular training for ablation (RGB protection), including RAW2RAW and localization (train)
- mode=4: OSN performance (val)
- mode=5: train a ISP using restormer for validation (train)
- mode=6: train passive image manipulation detection networks (train)
### Issues
- restormer cannot be loaded simultaneously with OSN network, because they share the same variable ```localizer```
- now you need to specify the path where each model locates

### How to test OSN in the code?

- run bash ./run_ISP_OSN.sh (mode==4)
- ~~Line 141 of Modified_invISP.py, modify the model as that of OSN network~~
- ~~specify Line 1319-2323 which provides the tamper source and mask~~
- the setting file is train_ISP_OSN.yml. If you want to do automatic copy-move, set ```inference_tamper_index=2``` and ```inference_load_real_world_tamper=False```
- ```using_which_model_for_test``` decides using which model for testing. ```discriminator_mask``` is our method, ```localizer``` is OSN.
- The average F1 score will be printed in the console.
- The main loop loops over the training set. Therefore you should manually kill the process when all the validate images are runned.

- Voila! the flow is optimize_parameters_router -> get_performance_of_OSN

### How to Test the baseline (RGB protection)?

- run bash ./run_ISP_OSN.sh (mode==4)
- the setting file is train_ISP_OSN.yml. Set ```test_baseline: true``` and ```task_name_customized_model: ISP_alone, load_customized_models: 64999```(which loads the trained baseline model from that location)
- Voila! the flow is optimize_parameters_router -> get_performance_of_OSN

### How to use Restormer during inference?
- whatever the option file yml is, set ```test_restormer: true```, and the model ```localizer``` will be Restormer. 
- Note: you cannot use OSN and Restormer at once!

### Logs
#### 1003

- test is now ok. Use ```mode=0```, which would contain protected image generation and tampering localization.

#### 1012
- update option and introduce base option

#### 1024
- the project has been restarted.

