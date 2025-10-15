
# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% a100
# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% a100 [21:09<00:00,  3.34it/s]
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78% rtx8000 30min around
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78% rtx8000 30min around
# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% v100 30min around

### threshold =200, topk=3 default
# v_13 Total: 4241, Correct: 3060, Accuracy: 72.15%, IMG-Accuracy: 66.93%
# v_20 Total: 4241, Correct: 3112, Accuracy: 73.38%, IMG-Accuracy: 69.51%
# v_21 Total: 4241, Correct: 3138, Accuracy: 73.99%, IMG-Accuracy: 70.80%
# top1 below
# t4 Total: 4241, Correct: 3046, Accuracy: 71.82%, IMG-Accuracy: 70.95%
# t10 Total: 4241, Correct: 3127, Accuracy: 73.73%, IMG-Accuracy: 71.69%
# t20 Total: 4241, Correct: 3180, Accuracy: 74.98%, IMG-Accuracy: 72.78%
# top2 with max token fixed
# t4 Total: 4241, Correct: 0, Accuracy: 0.00%, IMG-Accuracy: 0.00%
# max_token = 20 for this
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78%


# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% a100
# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% a100 [21:09<00:00,  3.34it/s]
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78% rtx8000 30min around
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78% rtx8000 30min around
# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% v100 30min around

### threshold =200, topk=3 default
# v_13 Total: 4241, Correct: 3060, Accuracy: 72.15%, IMG-Accuracy: 66.93%
# v_20 Total: 4241, Correct: 3112, Accuracy: 73.38%, IMG-Accuracy: 69.51%
# v_21 Total: 4241, Correct: 3138, Accuracy: 73.99%, IMG-Accuracy: 70.80%
# top1 below
# t4 Total: 4241, Correct: 3046, Accuracy: 71.82%, IMG-Accuracy: 70.95%
# t10 Total: 4241, Correct: 3127, Accuracy: 73.73%, IMG-Accuracy: 71.69%
# t20 Total: 4241, Correct: 3180, Accuracy: 74.98%, IMG-Accuracy: 72.78%
# top2 with max token fixed
# t4 Total: 4241, Correct: 0, Accuracy: 0.00%, IMG-Accuracy: 0.00%
# max_token = 20 for this
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78%

# |ScienceQA results|  IMG  |
# | Llava-1.5       | 72.68%|  2017/4241
# | Llava-1.5       | 72.78%|  max_token=20
# | Llava-1.5       | 72.78%|  max_token=3
# | visual model    |  IMG  |  Channel 542 replace mean
# | x9 y5 Wout w/b  | 72.93%|  layer 0-12
# | x9 y5 Wout w/b  | 72.14%|  layer 0-23
# | visual model    |  IMG  |  Channel 121, 407, 542, 711 replace mean
# | x9 y5 Wout w/b  | 72.09%|  layer 0-23 zero
# | x9 y5 Wout w/b  | 71.94%|  layer 0-23 mean
# | fusion model    |  IMG  |  Channel 3050 ,4727 ,4743 ,2772 ,1843 replace mean
# | x9 y5 Wout w    | 71.89%|  layer 0-39 zero
# | x9 y5 Wout w    | 71.84%|  layer 0-39 mean
# | whole model     |  IMG  |
# | x9 y5 Wout w/b  | 70.30%|  the above two setting combined (zero)
# | x9 y5 Wout w/b  | 70.15%|  replace with mean



# |   All  | visual model    |  IMG  | norm1in x1 removal |top3 threshold = 200
# | 72.15% | v_13            | 66.93%|  
# | 73.38% | v_20            | 69.51%|  
# | 73.99% | v_21            | 70.80%|  
# |   All  | fusion  model   |  IMG  | 
# | 71.82% | t_4             | 70.95%|  top1
# | 73.73% | t_10            | 71.69%|  top1
# | 74.98% | t_20            | 72.78%|  top1
# | 0%     | t_4             | 0%    |  top3 + max_token=3 30min
# | 0%     | t_10            | 0%    |  top3 + max_token=3
# | 17.21% | t_20            | 35.40%|  top3 + max_token=3
# | 0%     | t_4             | 0%    |  top3 + max_token=20 60min
# |   All  |ScienceQA results|  IMG  |
# | 74.89% | Llava-1.5       | 72.68%|  2000/4000s
# | 74.94% | Llava-1.5       | 72.78%|  max_token=20
# | 74.86% | Llava-1.5       | 72.78%|  max_token=3
# |   All  | visual model    |  IMG  |  y5 removal |top3 threshold = 200
# | xxxxx% | layer v_11      | 72.73%|  max_token=20
# | xxxxx% | layer v_12      | 72.09%|  max_token=20
# | xxxxx% | layer v_11,12   | 68.91%|  max_token=20
# | xxxxx% | layer v_11,12   | 69.16%|  max_token=20 top10
# | xxxxx% | layer v_11,12   | 68.67%|  max_token=20 top10 >20 replace
# |   All  | fusion  model   |  IMG  | 
# | 71.82% | t_3             | 00.00%| 
# | 71.82% | t_3             | 71.00%| top1
# |   All  | visual model    |  IMG  | norm2in y2 removal |top3 threshold = 200
# | xxxxx% | v_11            | 72.78%|  
# | xxxxx% | v_12            | 73.03%|  




# llava mag
# torch.Size([1, 969, 5120])
# norm1in  after change max at batch0: 1395.0
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.294921875

# llava mag replace x1 layer4 top1
# norm1in  before change max at batch0: 1395.0
# torch.Size([1, 969, 5120])
# norm1in  after change max at batch0: 830.0
# norm1in  before change max at batch0: 2.294921875
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.294921875

# llava mag replace x1 layer4 top2
# still similar mag but goes on forever
# norm1in  before change max at batch0: 1395.0
# torch.Size([1, 969, 5120])
# norm1in  after change max at batch0: 78.375
# norm1in  before change max at batch0: 3.265625
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 3.265625
# norm1in  before change max at batch0: 2.892578125
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.892578125
# norm1in  before change max at batch0: 2.30859375
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.30859375
# norm1in  before change max at batch0: 2.1171875
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.1171875

# llava mag replace x1 layer4 top10
# norm1in  before change max at batch0: 1395.0
# torch.Size([1, 969, 5120])
# norm1in  after change max at batch0: 55.21875
# norm1in  before change max at batch0: 3.265625
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 3.265625
# norm1in  before change max at batch0: 2.892578125
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.892578125
# norm1in  before change max at batch0: 2.30859375
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.30859375
# norm1in  before change max at batch0: 2.1171875
# torch.Size([1, 1, 5120])
# norm1in  after change max at batch0: 2.1171875



# RR
############ before [4, 39]
# x1  before change max at batch0: 1394.0
# x1  after change max at batch0: 1394.0, 0
# nextlayer x1 before restore max at batch0: 547.5, 0
# nextlayer x1 after restore max at batch0: 547.5
# x1  before change max at batch0: 2.54296875
# x1  after change max at batch0: 2.54296875, 0
# nextlayer x1 before restore max at batch0: 96.1875, 0
# nextlayer x1 after restore max at batch0: 96.1875


# x1  before change max at batch0: 1394.0
# x1  after change max at batch0: 72.9375, 3
# nextlayer x1 before restore max at batch0: 370.25, 3
# nextlayer x1 after restore max at batch0: 1024.0
# x1  before change max at batch0: 3.267578125
# x1  after change max at batch0: 3.267578125, 3
# nextlayer x1 before restore max at batch0: 76.3125, 0
# nextlayer x1 after restore max at batch0: 76.3125
# x1  before change max at batch0: 2.892578125
# x1  after change max at batch0: 2.892578125, 3
# nextlayer x1 before restore max at batch0: 72.0625, 0
# nextlayer x1 after restore max at batch0: 72.0625
# x1  before change max at batch0: 2.30859375
# x1  after change max at batch0: 2.30859375, 3
# nextlayer x1 before restore max at batch0: 76.9375, 0
# nextlayer x1 after restore max at batch0: 76.9375
