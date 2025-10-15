### 1. profile and modify in architecture.
# refer to .py files

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
################### load stored activation from profile models
activation_path = ''
device = 'cpu'
activations = torch.load(activation_path, map_location = device)
weight_path = ''
weights = torch.load(weight_path, map_location = device)
# weights = load_file(weight_path, device=device)

### 2. top magnitude plot
################### keys for each
branch = 'visual'
keys = ['norm1_in', 'norm1_out', 'attnq_proj', 'attnk_proj', 'attnv_proj', 'attn_score', 'attnout_proj',\
               'norm2_in', 'norm2_out', 'fc1', 'gelu', 'fc2']
titles = ['x1', 'x2', \
          'x3', 'x4', 'x5', \
          'x8', 'x9', \
          'y1', 'y2',\
           'y3', 'y4', 'y5', \
        ]
branch = 'text'
keys = ['norm1_in', 'norm1_out', 'attn_score', 'attnout_proj',\
               'norm2_in', 'norm2_out', 'fc11', 'fc1', 'silu', 'fc2_in', 'fc2_out']
titles = ['x1', 'x2', \
          'x8', 'x9', \
          'y1', 'y2',\
           'y3', 'y4', 'y5',\
            'y6', 'y7', \
        ]
################### top plot
num_vit_layers = 24
num_text_layers = 40
top1_activations = []
top2_activations = []
top3_activations = []
median_activations = []
def select_key_cal_topk_andmedian(keys, absmean=False, batchid=None, branch='visual'):
    num_layers = num_vit_layers if branch=='visual' \
    else num_text_layers
    for id in range(num_layers):
        for k in keys:
            key_for_act = f'llava.visual.block_{id}.'+k if branch == 'visual' \
            else f'llava.text.block_{id}.'+k
            if batchid is not None:
                activation_target = activations[key_for_act][batchid].abs()
            else:
                if absmean:
                    activation_target = activations[key_for_act].abs().mean(dim=0)
                else:
                    activation_target = activations[key_for_act].mean(dim=0).abs()
            sorted_target, _ = torch.sort(activation_target.flatten(), descending=True)
            top1_activations.append(sorted_target[0].numpy())
            top2_activations.append(sorted_target[1].numpy())
            top3_activations.append(sorted_target[2].numpy())
            median_activations.append(torch.median(activation_target).numpy())

############################### set args for plot top ######################################
batchid = None
branch = 'visual'
keys = []
titles = []
############################### set args for plot top ######################################

select_key_cal_topk_andmedian(keys, False, batchid, branch)

x_ = np.arange(len(top1_activations))

plt.figure(figsize=(20, 6))
plt.plot(x_, top1_activations, label='Top-1', marker='o')
plt.plot(x_, top2_activations, label='Top-2', marker='o')
plt.plot(x_, top3_activations, label='Top-3', marker='o')
plt.plot(x_, median_activations, label='Median', marker='v')
# massive = [act * 1000 for act in median_activations]
# plt.plot(x_, massive, label='massive threshold', marker='v')

# Add layer labels
plt.xticks(np.arange(num_vit_layers)*len(keys), np.arange(num_vit_layers), fontsize=10)
plt.xlabel('Layer id')
plt.ylabel(f'Llava {branch} top Magnitude-{titles[-1]}', fontsize=15)
plt.legend(ncol=2, fontsize=15)

### 3. 3d plot
################### 3.1 3d MA plot
def plot_3d_MA_subplots_across_block(activations, batch_id=None, block_ids=0, \
                                   title="Activation Visualization", token_len = None, start_id=0, branch='visual'\
                                    ,start_c=0,token_lenc = None, modify=False, output_channel=False, output_token=False, had=False):
    # batch_size = activation.size(0)
    # activation = activation.abs().numpy()
    plot_size = len(keys) * len(block_ids)
    rows = (plot_size + 3) // 4  # Compute the number of rows needed (4 subplots per row)
    fig = plt.figure(figsize=(20, 6 * rows))
    j = -1
    for block_id in block_ids:
        j += 1
        for i in range(len(keys)):
            key_for_act = f'llava.visual.block_{block_id}.{keys[i]}' if branch=='visual' \
                                    else f'llava.text.block_{block_id}.{keys[i]}' # switch between 
            
            if had:
                activation = activations[key_for_act][batch_id].float().clone() if batch_id is not None else activations[key_for_act].mean(dim=0).float().clone()
                Had = hadamard_matrix(activation.shape[-1]).float()/torch.sqrt(torch.tensor(activation.shape[-1], dtype=torch.float))
                activation = torch.matmul(activation, Had).abs()
            else:
                activation = activations[key_for_act][batch_id].abs().float() if batch_id is not None else activations[key_for_act].mean(dim=0).abs().float()
            
            if modify:
                mean = torch.mean(activation.flatten())
                median = torch.median(activation.flatten())
                zero = torch.zeros_like(mean)
                topk_values, topk_indices = torch.topk(activation.flatten(), 3)
                topk_indices = torch.unravel_index(topk_indices, activation.shape)
                for value, indice in zip(topk_values, zip(*topk_indices)):
                    # if value > 100 * median and value>20:
                    if value>20:
                        activation[indice] = median
            print(f"{block_id}, K:{kurtosis(activation)}")
            if output_channel:
                act_std = activation.std()
                act_mean = activation.mean()
                _, dim = activation.shape
                outlier_channel = []
                for c_dim in range(dim):
                    actc_std = activation[:, c_dim].std()
                    actc_mean = activation[:, c_dim].mean()
                    if actc_mean > act_mean + ratio * act_std and actc_std < threshold:
                        outlier_channel.append(c_dim)
                print(block_id, outlier_channel)
            if output_token:
                act_std = activation.std()
                act_mean = activation.mean()
                dim, _ = activation.shape
                outlier_token = []
                for t_dim in range(dim):
                    actt_std = activation[t_dim, :].std()
                    actt_mean = activation[t_dim, :].mean()
                    if actt_mean > act_mean + ratio * act_std and actt_std < threshold:
                        outlier_token.append(t_dim)
                print(block_id, outlier_token)
            if token_len is None:
                token_len = activation.shape[0]
            if token_lenc is None:
                token_lenc = activation.shape[1]
            if start_c + token_lenc > activation.shape[1]:
                start_c = activation.shape[1] - token_lenc
                print(f'channel idx overflow, clamp to{start_c}')
            if start_id + token_len > activation.shape[0]:
                start_id = activation.shape[0] - token_len
                print(f'token idx overflow, clamp to{start_id}')
            ax = fig.add_subplot(rows, 4, i + 1 + len(keys)*j, projection='3d')
            x = np.arange(token_lenc) + start_c  # Channels (C)
            y = np.arange(token_len) + start_id  # Sequence length (N)
            x, y = np.meshgrid(x, y)
            z = activation[start_id:token_len+start_id, start_c:start_c+token_lenc].numpy()  # Use the clipped activation with transposing
            # print(block_id, keys[i], z.max(), z.max()/np.median(z), (z.max()==activation.numpy()).nonzero())
            ax.plot_wireframe(x, y, z, color="royalblue", cstride=0, linewidth=1.5) # royalblue
            ax.set_title(f"{titles[i]}-Layer{block_id}",fontsize=20)
            ax.set_xlabel('Channels (C)')
            ax.set_ylabel('Sequence Length (N)')
            ax.set_zlabel('Activation Magnitude')
        # ax.set_zlim(0,20) # this can change

    plt.tight_layout()
    plt.show()

################### 3.2 3d channel wise plot
def plot_3d_activation_subplots_across_block(activations, batch_id=None, block_ids=0, \
                                   title="Activation Visualization", token_len = None, start_id=0, branch='visual'\
                                    ,start_c=0,token_lenc = None, modify=False, output_channel=False, output_token=False, had=False):
    # batch_size = activation.size(0)
    # activation = activation.abs().numpy()
    plot_size = len(keys) * len(block_ids)
    rows = (plot_size + 3) // 4  # Compute the number of rows needed (4 subplots per row)
    fig = plt.figure(figsize=(20, 6 * rows))
    j = -1
    for block_id in block_ids:
        j += 1
        for i in range(len(keys)):
            key_for_act = f'llava.visual.block_{block_id}.{keys[i]}' if branch=='visual' \
                                    else f'llava.text.block_{block_id}.{keys[i]}' # switch between 
            
            if had:
                activation = activations[key_for_act][batch_id].float().clone() if batch_id is not None else activations[key_for_act].mean(dim=0).float().clone()
                Had = hadamard_matrix(activation.shape[-1]).float()/torch.sqrt(torch.tensor(activation.shape[-1], dtype=torch.float))
                activation = torch.matmul(activation, Had).abs()
            else:
                activation = activations[key_for_act][batch_id].abs().float() if batch_id is not None else activations[key_for_act].mean(dim=0).abs().float()
            
            if modify:
                mean = torch.mean(activation.flatten())
                median = torch.median(activation.flatten())
                zero = torch.zeros_like(mean)
                topk_values, topk_indices = torch.topk(activation.flatten(), 3)
                topk_indices = torch.unravel_index(topk_indices, activation.shape)
                for value, indice in zip(topk_values, zip(*topk_indices)):
                    # if value > 100 * median and value>20:
                    if value>20:
                        activation[indice] = median
            print(f"{block_id}, K:{kurtosis(activation)}")
            if output_channel:
                act_std = activation.std()
                act_mean = activation.mean()
                _, dim = activation.shape
                outlier_channel = []
                for c_dim in range(dim):
                    actc_std = activation[:, c_dim].std()
                    actc_mean = activation[:, c_dim].mean()
                    if actc_mean > act_mean + ratio * act_std and actc_std < threshold:
                        outlier_channel.append(c_dim)
                print(block_id, outlier_channel)
            if output_token:
                act_std = activation.std()
                act_mean = activation.mean()
                dim, _ = activation.shape
                outlier_token = []
                for t_dim in range(dim):
                    actt_std = activation[t_dim, :].std()
                    actt_mean = activation[t_dim, :].mean()
                    if actt_mean > act_mean + ratio * act_std and actt_std < threshold:
                        outlier_token.append(t_dim)
                print(block_id, outlier_token)
            if token_len is None:
                token_len = activation.shape[0]
            if token_lenc is None:
                token_lenc = activation.shape[1]
            ax = fig.add_subplot(rows, 4, i + 1 + len(keys)*j, projection='3d')
            x = np.arange(token_lenc) + start_c  # Channels (C)
            y = np.arange(token_len) + start_id  # Sequence length (N)
            x, y = np.meshgrid(x, y)
            z = activation[start_id:token_len+start_id, start_c:start_c+token_lenc].numpy()  # Use the clipped activation with transposing
            # print(block_id, keys[i], z.max(), z.max()/np.median(z), (z.max()==activation.numpy()).nonzero())
            ax.plot_surface(x, y, z, cmap='coolwarm') # viridis
            ax.set_title(f"{titles[i]}-Layer{block_id}",fontsize=20)
            ax.set_xlabel('Channels (C)')
            ax.set_ylabel('Sequence Length (N)')
            ax.set_zlabel('Activation Magnitude')
        # ax.set_zlim(0,20) # this can change

    plt.tight_layout()
    plt.show()

############################### set args for plot below ######################################
branch = 'visual'
keys = ['norm1_in', 'norm1_out', 'attnq_proj', 'attnk_proj', 'attnv_proj', 'attn_score', 'attnout_proj',\
               'norm2_in', 'norm2_out', 'fc1', 'gelu', 'fc2']
titles = ['x1', 'x2', \
          'x3', 'x4', 'x5', \
          'x8', 'x9', \
          'y1', 'y2',\
           'y3', 'y4', 'y5', \
        ]
keys = ['norm1_in']
titles = ['x1']
block_ids = [_ for _ in range(24)]
token_len = None
start_id =  0       # zoomed in 3d plot based on top inspection
token_lenc = None
start_c    = 0    
ratio=3
threshold=1
############################### set args for plot above ######################################
plot_3d_activation_subplots_across_block(activations, 0, block_ids, token_len=token_len, start_id=start_id \
                                         ,start_c=start_c, token_lenc=token_lenc, branch=branch) # , output_channel=True , output_token=True, modify=True



######### 3.3 layernorm break down
def plot_3d_activation_subplots_forNorm(activations, batch_id=None, block_ids=0, \
                                   title="Activation Visualization", token_len = None, start_id=0, branch='visual'\
                                    ,start_c=0,token_lenc = None, modify=False, output_channel=False, output_token=False):
    # batch_size = activation.size(0)
    # activation = activation.abs().numpy()
    plot_size = len(keys) * len(block_ids)
    rows = (plot_size + 2) // 3  # Compute the number of rows needed (4 subplots per row)
    fig = plt.figure(figsize=(20, 6 * rows))
    j = -1
    for block_id in block_ids:
        j += 1
        for i in range(len(keys)):
            if 'mid' in keys[i]:
                key_for_act = f'llava.visual.block_{block_id}.{keys[i-1]}' if branch=='visual' \
                                        else f'llava.text.block_{block_id}.{keys[i-1]}' # switch between 
                
                activation = activations[key_for_act][batch_id].clone() if batch_id is not None else activations[key_for_act].mean(dim=0).clone()
                import torch.nn as nn
                ln = nn.LayerNorm(activation.shape[-1],elementwise_affine=False)
                activation = ln(activation).abs().float()            
            else:
                key_for_act = f'llava.visual.block_{block_id}.{keys[i]}' if branch=='visual' \
                                        else f'llava.text.block_{block_id}.{keys[i]}' # switch between 
                
                activation = activations[key_for_act][batch_id].abs().float() if batch_id is not None else activations[key_for_act].mean(dim=0).abs().float()
            if modify:
                mean = torch.mean(activation.flatten())
                median = torch.median(activation.flatten())
                zero = torch.zeros_like(mean)
                topk_values, topk_indices = torch.topk(activation.flatten(), 10)
                topk_indices = torch.unravel_index(topk_indices, activation.shape)
                for value, indice in zip(topk_values, zip(*topk_indices)):
                    if value > 300 * median:
                        activation[indice] = median
            if output_channel:
                act_std = activation.std()
                act_mean = activation.mean()
                _, dim = activation.shape
                outlier_channel = []
                for c_dim in range(dim):
                    actc_std = activation[:, c_dim].std()
                    actc_mean = activation[:, c_dim].mean()
                    if actc_mean > act_mean + ratio * act_std and actc_std < threshold:
                        outlier_channel.append(c_dim)
                print(block_id, 'outlier_channel:', outlier_channel)
            if output_token:
                act_std = activation.std()
                act_mean = activation.mean()
                dim, _ = activation.shape
                outlier_token = []
                for t_dim in range(dim):
                    actt_std = activation[t_dim, :].std()
                    actt_mean = activation[t_dim, :].mean()
                    if actt_mean > act_mean + ratio * act_std and actt_std < threshold:
                        outlier_token.append(t_dim)
                print(block_id, 'outlier_token:',outlier_token)
            if token_len is None:
                token_len = activation.shape[0]
            if token_lenc is None:
                token_lenc = activation.shape[1]
            ax = fig.add_subplot(rows, 3, i + 1 + len(keys)*j, projection='3d')
            x = np.arange(token_lenc) + start_c  # Channels (C)
            y = np.arange(token_len) + start_id  # Sequence length (N)
            x, y = np.meshgrid(x, y)
            z = activation[start_id:token_len+start_id, start_c:start_c+token_lenc].numpy()  # Use the clipped activation with transposing
            # print(block_id, keys[i], z.max(), z.max()/np.median(z), (z.max()==activation.numpy()).nonzero())
            ax.plot_surface(x, y, z, cmap='coolwarm') # viridis 
            ax.set_title(f"{titles[i]}-Layer{block_id}", fontsize=20)
            ax.set_xlabel('Channels (C)')
            ax.set_ylabel('Sequence Length (N)')
            ax.set_zlabel('Activation Magnitude')
        # ax.set_zlim(0,20) # this can change

    plt.tight_layout()
    plt.show()
############################### set args for plot  ######################################
branch = 'visual'
keys = ['norm1_in','mid', 'norm1_out']
titles = ['x1','norm_only', 'x2']
# keys = ['norm2_in','mid', 'norm2_out']
# titles = ['y1','norm_only', 'y2']
block_ids = [0, 4, 8, 11, 12, 13, 16, 20, 23] # should refer to top results. so we know where to inspect
block_ids = [_ for _ in range(24)]
# block_ids = [0, 6, 11]
block_ids = [10, 18, 23]
token_len = None
start_id  = 0     # zoomed in 3d plot based on top inspection
token_lenc = None
start_c    = 0     # 
ratio=3
threshold=5
############################### set args for plot  ######################################
plot_3d_activation_subplots_forNorm(activations, 0, block_ids, token_len=token_len, start_id=start_id \
                                         ,start_c=start_c, token_lenc=token_lenc, branch=branch, \
                                            output_channel=True)# output_channel=True, output_token=True




######### 3.4 linear weight visualization
def plot_3d_weight_channel_acrossblock(weights, block_ids, out_len=None, start_id=0, in_len=None, instart_id=0):
    plot_size = len(keys) * len(block_ids)
    rows = (plot_size + 3) // 4  # Compute the number of rows needed (4 subplots per row)
    fig = plt.figure(figsize=(24, 6 * rows))
    j = -1
    for block_id in block_ids:
        j += 1
        for i in range(len(keys)):
            out_len_ = out_len
            in_len_ = in_len # handle different dim
            weight = weights['vision_model.encoder.layers.'+f'{block_id}.{keys[i]}'+'.weight']
            in_channel = weight.shape[1]
            out_channel = weight.shape[0]
            if out_len_ is None:
                out_len_ = out_channel
            if in_len_ is None:
                in_len_ = in_channel
            x = np.arange(out_len_) + start_id
            y = np.arange(in_len_) + instart_id
            
            y, x = np.meshgrid(y, x)
            ax = fig.add_subplot(rows, 4, i + 1 + j*len(keys), projection='3d')
            weight_target = weight[start_id:start_id+out_len_, instart_id:instart_id+in_len_].abs()
            ax.plot_surface(x, y, weight_target, cmap='coolwarm')
            # ax.plot_wireframe(x, y, weight_target, cmap='coolwarm', rstride=0, linewidth=0.5)
            ax.set_title(f'{titles[i]}-weight Layer{block_id}', fontsize=20)
            ax.set_xlabel('out_channel')
            ax.set_ylabel('in_channel')
            ax.set_zlabel('weight magnitude')
    plt.tight_layout()
    plt.show()
#################### input
keys = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj',\
        'self_attn.out_proj','mlp.fc1', 'mlp.fc2']
titles = ['q','k','v','out','fc1','fc2']
keys = ['self_attn.out_proj']
titles = ['out']
block_ids = [0, 1, 2, 3, 11, 12, 13, 14, 19, 20, 21, 23]
block_ids = [9]
out_len =    None      #15
start_id =   0      #   535
in_len =     None
instart_id = 0
#################### input
plot_3d_weight_channel_acrossblock(weights, block_ids \
     ,out_len=out_len, start_id=start_id,in_len=in_len, instart_id=instart_id,)









######### 3.4 layernorm weight visualization

def plot_2d_ln_acrossblock(weights, block_ids, outlier_id=0, label='None'):
    plot_size = len(keys) * len(block_ids)
    # rows = plot_size  # Compute the number of rows needed (4 subplots per row)
    rows = (plot_size + 1) // 2
    fig = plt.figure(figsize=(24, 3 * rows))
    j = -1
    for block_id in block_ids:
        j += 1
        for i in range(len(keys)):
            weight = weights['vision_model.encoder.layers.'+f'{block_id}.{keys[i]}']
            x = np.arange(len(weight))
            x_outlier = outlier_id
            ax = fig.add_subplot(rows, 2, i + 1 + j*len(keys))
            plt.axvline(x=x_outlier, color='gray', linestyle='dotted', linewidth=2, label=f'Outlier occur at {label} x={x_outlier}')# can set ymin and ymax as well
            ax.set_title(f'{titles[i]} Layer{block_id}', fontsize=20)
            ax.plot(x, weight, label='bias')
            ax.set_xlabel('channel dimension')
            ax.set_ylabel(f'{titles[i]} value')
    plt.tight_layout()
    plt.show()

keys = ['layer_norm1.weight', 'layer_norm1.bias']
titles = ['ln1_weight','ln1_bias']
keys = ['layer_norm2.weight', 'layer_norm2.bias']
titles = ['ln2_weight','ln2_bias']
block_ids = [0, 1, 2, 3, 11, 12, 13, 14, 19, 20, 21, 23]
block_ids = [_ for _ in range(24)]

plot_2d_ln_acrossblock(weights, block_ids, outlier_id=None)


######### utils
def hadamard_matrix(n):
    """Constructs a Hadamard matrix of size n x n, where n is a power of 2."""
    if n == 1:
        return torch.tensor([[1]])
    
    # Recursively construct the Hadamard matrix
    H = hadamard_matrix(n // 2)
    return torch.cat(
        [
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ],
        dim=0,
    )
def kurtosis(x):
    """Calculates the kurtosis of a tensor."""
    ### reflect the outlier situation, used in spinquant
    ### in our implementation, we calculate activation in abs()
    mean = torch.mean(x)
    std = torch.std(x)
    z = (x - mean) / std  # Standardize the data
    return torch.mean(z**4)