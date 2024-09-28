import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import time as time


import os
import wandb
import argparse
from tqdm import tqdm
from npy_models import AVQA
from npy_utils import AVQADataset
from npy_plot import create_report

import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter         


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from transformers.utils import logging
logging.set_verbosity_error() 


TIMESTAMP = "{0:%m-%d_%H:%M/}".format(datetime.datetime.now()) 
SEED=42
# writer = SummaryWriter()
os.environ['WANDB_API_KEY'] = 'your_api_key'

# Initialize Wandb with the new API key
wandb.login()

torch.manual_seed(SEED)

rank = torch.cuda.current_device()
world_size = torch.cuda.device_count()

parser = argparse.ArgumentParser()

parser.add_argument('--ddp', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--kl', default=0.0, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--ep', default=50, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_norm', default=8.0, type=float)
args = parser.parse_args()

# Define Sweep Configuration
project_name = "avqa_aqvq"
sweep_config = {
    'method': 'grid',  # or 'random' for random search
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.01]
        },
        'batch_size': {
            'value': 16
        },
        'epochs': {
            'value': 50
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project=project_name)



plot_dict = {
    
    'train_losses': [],
    'train_accs': [],
    'val_losses': [],
    'val_accs': [],

    'a_countings': [],
    'a_comparatives': [],
    'a_avgs': [],

    'v_countings': [],
    'v_locations': [],
    'v_avgs': [],

    'av_existentials': [],
    'av_locations': [],
    'av_countings': [],
    'av_comparatives': [],
    'av_temporals': [],
    'av_avgs': []

}

def aggregate_metrics(local_metric):
    
    metric_tensor = torch.tensor(local_metric)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    return metric_tensor.item() / world_size


def train(epoch, n_epochs, rank, model, criterion, train_loader, optimizer, scheduler):
    
    model.train()
    
    train_loss = 0
    train_corrects = 0
    train_samples = 0
    train_load_time = time.time()
    
    for batch_idx, sample in enumerate(train_loader):

        optimizer.zero_grad()
        # print(f'Loaded in: {((time.time() - train_load_time)):.2f} seconds')
        init_time = time.time()
        
        # Load
        a, v, q, mask, ans = sample['audio'].cuda(rank), sample['visual'].cuda(rank), \
            sample['input_ids'].cuda(rank), sample['attn_mask'].cuda(rank), sample['answer'].cuda(rank)
        
        # Forward
        av_loss, ag_loss, pred = model(a, v, q, mask)
        # loss = av_loss * 0.3 + criterion(pred, ans)
        ce_loss = criterion(pred, ans)
        # print(f'av_loss: {av_loss.item()}, ag_loss: {ag_loss.item()}, ce_loss: {loss.item()}')
        # loss = ce_loss + 0.1 * av_loss + ag_loss * 0.1
        loss = ce_loss
        loss.backward()
        
        # Backward
        # print('not used params: ', end='')
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name, end=', ')
        nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()

        # Logging
        train_loss += loss.item()
        train_corrects += torch.sum(pred.argmax(1)==ans.argmax(1), dtype=torch.float16).item()
        train_samples += a.size(0)

        # if batch_idx == 100:
            # print(f"Rank: {rank} | Epoch: {epoch} / {n_epochs} | Batch: {batch_idx+1} / {len(train_loader)} | Loss: {(loss.item()):.4f} | Accuracy: {(train_corrects / train_samples):.4f}", end=' | ', flush=True)
            # print(f"AV_Loss: {(av_loss):.4f} | AG_Loss: {(ag_loss):.4f}", flush=True)
    
        avg_loss = train_loss/len(train_loader)
        train_acc = train_corrects/train_samples
 
    # plot_dict['train_losses'].append(train_loss/len(train_loader))
    # plot_dict['train_accs'].append(train_corrects/train_samples)
    # print(f"Epoch: {epoch} / {n_epochs} | Train_loss: {(train_loss / len(train_loader)):.4f} | Train_acc: {(train_corrects / train_samples):.4f} | ", end='')
    return avg_loss, train_acc, av_loss, ag_loss
    # writer.add_scalars('Loss/train', {'train': train_loss / len(train_loader)}, epoch * len(train_loader))
    # writer.add_scalars('Accuracy/train', {'train': train_acc / train_samples}, epoch * len(train_loader))
    
    

def eval(epoch, rank, model, criterion, val_loader):
    
    model.eval()
    
    val_loss = 0
    val_corrects = 0
    val_samples = 0
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):

            a, v, q, mask, ans = sample['audio'].cuda(rank), sample['visual'].cuda(rank), \
                sample['input_ids'].cuda(rank), sample['attn_mask'].cuda(rank), sample['answer'].cuda(rank)

            av_loss, ag_loss, pred = model(a, v, q, mask)
            # loss = av_loss * 0.3 + criterion(pred, ans)
            ce_loss = criterion(pred, ans)
            # print(f'av_loss: {av_loss.item()}, ag_loss: {ag_loss.item()}, ce_loss: {loss.item()}')
            # loss = ce_loss + 0.1 * av_loss + ag_loss * 0.1
            loss = ce_loss

            val_loss += loss.item()
            val_corrects += torch.sum(pred.argmax(1)==ans.argmax(1), dtype=torch.float16).item()
            val_samples += a.size(0)

    
    avg_loss = val_loss/len(val_loader)
    val_acc = val_corrects/val_samples
    # print(f"Val_loss: {(val_loss / len(val_loader)):.4f} | Val_acc: {(val_acc):.4f}")
    # writer.add_scalars('Loss/val', {'val': val_loss / len(val_loader)}, epoch * len(train_loader))
    # writer.add_scalars('Accuracy/val', {'train': val_acc / val_samples}, epoch * len(train_loader))

    return avg_loss, val_acc



def test(rank, model, test_loader):

    question_types = {'Audio': {'Counting': [], 'Comparative': []}, 'Visual': {'Counting': [], 'Location': []},
                      'Audio-Visual': {'Existential': [], 'Location': [], 'Counting': [], 'Comparative': [], 'Temporal': []}}
    
    labels_dict = {'electric_bass': 0, 'middle': 1, 'trumpet': 2, 'pipa': 3, 'one': 4, 'acoustic_guitar': 5,\
                    'cello': 6, 'four': 7, 'xylophone': 8, 'nine': 9, 'accordion': 10, 'left': 11, \
                    'simultaneously': 12, 'seven': 13, 'eight': 14, 'banjo': 15, 'suona': 16, \
                    'two': 17, 'tuba': 18, 'ukulele': 19, 'congas': 20, 'piano': 21, 'bagpipe': 22,\
                    'clarinet': 23, 'guzheng': 24, 'no': 25, 'violin': 26, 'three': 27, \
                    'right': 28, 'six': 29, 'yes': 30, 'drum': 31, 'five': 32, 'bassoon': 33, \
                    'zero': 34, 'more than ten': 35, 'saxophone': 36, 'flute': 37, 'outdoor': 38, \
                    'erhu': 39, 'indoor': 40}
    total_q = 0
    n_corrects = 0
    preds = []
    trues = []
    target_names = []
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):

            a, v, q, mask, ans = sample['audio'].cuda(rank), sample['visual'].cuda(rank),\
                sample['input_ids'].cuda(rank), sample['attn_mask'].cuda(rank), sample['answer'].cuda(rank)
    
            _, _, pred = model(a, v, q, mask)
            
            modality_type = sample['question_type'][0]
            question_type = sample['question_type'][1]

            pred = pred.argmax(dim=1)
            true = ans.argmax(dim=1)
            preds.append(pred.tolist())
            trues.append(true.tolist())

            for i in range(pred.size(0)):

                question_types[modality_type[i]][question_type[i]].append((pred[i]==true[i]))
                # target_names.append(sample['answer'][i])

            # Total accuracy
            total_q += pred.size(0)
            n_corrects += torch.sum(pred==true, dtype=torch.float16).item()
    
    a_counting = sum(question_types['Audio']['Counting']).item() / (len(question_types['Audio']['Counting'])+1)
    a_comparative = sum(question_types['Audio']['Comparative']).item() / (len(question_types['Audio']['Comparative'])+1)
    a_avg = (a_counting + a_comparative) / 2
    plot_dict['a_countings'].append(a_counting)
    plot_dict['a_comparatives'].append(a_comparative)
    plot_dict['a_avgs'].append(a_avg)
    print(f"A-Counting: {(a_counting):.4f} | ", end='', flush=True)
    print(f"A-Comparative: {(a_comparative):.4f} | ", end='', flush=True)
    print(f"A-Avg: {(a_avg):.4f}", flush=True)

    v_counting = sum(question_types['Visual']['Counting']).item() / (len(question_types['Visual']['Counting'])+1)
    v_location = sum(question_types['Visual']['Location']).item() / (len(question_types['Visual']['Location'])+1)
    v_avg = (v_counting + v_location) / 2
    plot_dict['v_countings'].append(v_counting)
    plot_dict['v_locations'].append(v_location)
    plot_dict['v_avgs'].append(v_avg)
    print(f"V-Counting: {(v_counting):.4f} | ", end='', flush=True)
    print(f"V-Location: {(v_location):.4f} | ", end='', flush=True)
    print(f"V-Avg: {(v_avg):.4f}", flush=True)

    av_existential = sum(question_types['Audio-Visual']['Existential']).item() / (len(question_types['Audio-Visual']['Existential'])+1)
    av_location = sum(question_types['Audio-Visual']['Location']).item() / (len(question_types['Audio-Visual']['Location'])+1)
    av_counting = sum(question_types['Audio-Visual']['Counting']).item() / (len(question_types['Audio-Visual']['Counting'])+1)
    av_comparative = sum(question_types['Audio-Visual']['Comparative']).item() / (len(question_types['Audio-Visual']['Comparative'])+1)
    av_temporal = sum(question_types['Audio-Visual']['Temporal']).item() / (len(question_types['Audio-Visual']['Temporal'])+1)

    plot_dict['av_existentials'].append(av_existential)
    plot_dict['av_locations'].append(av_location)
    plot_dict['av_countings'].append(av_counting)
    plot_dict['av_comparatives'].append(av_comparative)
    plot_dict['av_temporals'].append(av_temporal)
    av_avg = (av_existential + av_location + av_counting + av_comparative + av_temporal) / 5
    plot_dict['av_avgs'].append(av_avg)
    
    print(f"AV-Existential: {(av_existential):.4f} | ", end='', flush=True)
    print(f"AV-Location: {(av_location):.4f} | ", end='', flush=True)
    print(f"AV-Counting: {(av_counting):.4f} | ", end='', flush=True)
    print(f"AV-Comparative: {(av_comparative):.4f} | ", end='', flush=True)
    print(f"AV-Temporal: {(av_temporal):.4f} | ", end='', flush=True)
    print(f"AV-Avg: {(av_avg):.4f}", flush=True)
    
    print(f"Overall Accuracy: {(n_corrects / total_q):.4f}", flush=True)
    
    # print(labels_dict)
    y_pred = []
    for pred in preds:
        y_pred += pred
    y_true = []
    for true in trues:
        y_true += true
    print(classification_report(y_true, y_pred, zero_division=0.0), flush=True)

    return n_corrects / total_q


################################################ DDP ################################################
def setup_ddp(rank, world_size):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group('nccl', rank=rank, world_size=world_size)

def prepare_ds(ds_label, batch_size, rank, shuffle=False, pin_memory=False, num_workers=0):
    
    dataset = AVQADataset(ds_label)
    sampler = DistributedSampler(dataset, rank=rank, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            pin_memory=pin_memory, num_workers=num_workers, 
                            sampler=sampler)
    
    return dataloader

################################################ DDP ################################################
def main(rank):

    world_size = torch.cuda.device_count()
    model = AVQA().cuda(rank)
    # print(f'model paramters: {sum(p.numel() for p in model.parameters())}')
    with wandb.init(config=None):
        config = wandb.config
        if args.ddp and world_size > 1:

            setup_ddp(rank=rank, world_size=world_size)
            model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
            train_loader = prepare_ds('train', batch_size=config.batch_size, rank=rank)
            val_loader = prepare_ds('val', batch_size=config.batch_size, rank=rank)
            test_loader = prepare_ds('test', batch_size=config.batch_size, rank=rank)
        else:
            train_loader = DataLoader(AVQADataset('train'), batch_size=config.batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(AVQADataset('val'), batch_size=config.batch_size, num_workers=1, pin_memory=True)
            test_loader = DataLoader(AVQADataset('test'), batch_size=config.batch_size, num_workers=1, pin_memory=True)
        
        print(f'rank({rank}) - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}', flush=True)

        train_start = time.time()

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        n_epochs = args.ep
        epochs = range(n_epochs)
        test_accs = []

        for epoch in tqdm(epochs):

            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0
            # kl = args.kl
            if args.ddp:
                train_loader.sampler.set_epoch(epoch)
            t_loss, t_acc, av_loss, ag_loss = train(epoch+1, n_epochs, rank, model, criterion, train_loader, optimizer, scheduler)
            train_loss += t_loss
            train_acc += t_acc
            
            # if epoch % 10 == 0:
            #     scheduler.step()
            
            if rank == 0:
                v_loss, v_acc = eval(epoch, rank, model, criterion, val_loader)
                val_loss += v_loss
                val_acc += v_acc
                
                plot_dict['train_losses'].append(train_loss/world_size)
                plot_dict['train_accs'].append(train_acc/world_size)
                plot_dict['val_losses'].append(val_loss/world_size)
                plot_dict['val_accs'].append(val_acc/world_size)
                
                print(f'Epoch: {epoch+1}/{n_epochs} | AV_Loss: {(av_loss):.4f} | AG_Loss: {(ag_loss):.4f} | train_loss: {(train_loss/world_size):.4f} | train_acc: {(train_acc/world_size):.4f} | ', end='', flush=True)
                print(f'val_loss: {(val_loss/world_size):.4f} | val_acc: {(val_acc/world_size):.4f}', flush=True)        
                wandb.log({'train_loss': train_loss/world_size, 'epoch': epoch})
                wandb.log({'val_loss': val_loss/world_size, 'epoch': epoch})
                

        test_accs.append(test(rank, model, test_loader))

        # Plot
        # job_name = f'lr_{args.lr}_ep_{args.ep}'
        # main_path = f'runs/{TIMESTAMP}/{job_name}/figs'
        # os.makedirs(main_path, exist_ok=True)
        # create_report(train_start, main_path, epochs, plot_dict)

    if args.ddp:
        destroy_process_group()

    
def run_main():

    main(rank)

if __name__ == "__main__":

    """
    Include metric to show best result from each category
    """

    start_train = time.time()
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        args.ddp = False

    if args.ddp and world_size > 1:
        print("DDP enabled")
        mp.spawn(main, nprocs=world_size)
    else:
        # print("DDP disenabled")
        # main(rank)
        wandb.agent(sweep_id, function=run_main, count=10)

    print(f'Train ended in {(time.time() - start_train)//60} min')
