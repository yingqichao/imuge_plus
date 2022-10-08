# A framework for all papers
## DefensiveRAW: Robust Image Authentication via Transferable RAW Protection

### Unsolved issues

### notes

How to test OSN in the code
- run bash ./run_ISP_OSN.sh (mode==4)
- Line 141 of Modified_invISP.py, modify the model as that of OSN network
- specify Line 1319-2323 which provides the tamper source and mask
- the setting file is train_ISP_OSN.yml. If you want to do automatic copy-move, set ```inference_tamper_index=2``` and ```inference_load_real_world_tamper=False```

Output:
- The average F1 score will be printed in the console
- The main loop loops over the training set. Therefore you should manually kill the process when all the validate images are runned.


other notes:
- the flow is optimize_parameters_router -> get_performance_of_OSN
- use gpu 0 as default
- 

### Logs
#### 1003

- test is now ok. Use ```mode=0```, which would contain protected image generation and tampering localization.
