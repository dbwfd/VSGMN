import wandb
import torch
import torch.optim as optim
import pickle
import torch.nn as nn
import numpy as np
from model_vsgmn import VSGMN
from dataset import SUNDataLoader
from helper_func import eval_zs_gzsl

# init wandb from config file

wandb.init(project='GCN', config='wandb_config/sun_czsl_best.yaml')
config = wandb.config
print('Config file from wandb:', config)

# load dataset
dataloader = SUNDataLoader('.', "cuda:0")

# set random seed
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

init_w2v_att = dataloader.w2v_att
att = dataloader.att
att[att<0] = 0
normalize_att = dataloader.normalize_att

class_path = f'w2v/{config.dataset}_class.pkl'
# att_binary_paths = f'data/AWA2/Animals_with_Attributes2/predicate-matrix-binary.txt'

# batt=torch.from_numpy(np.loadtxt(att_binary_paths)).float().to(config.device)
# with open(class_path, 'rb') as f:
#     w2v_class = pickle.load(f)
# w2v_class = np.array(w2v_class)
# w2v_class = torch.from_numpy(w2v_class).float().to(config.device)
vp=dataloader.vp







#  model
model=VSGMN(config,att,init_w2v_att,vp,dataloader.seenclasses, dataloader.unseenclasses).to(config.device)

# pretrained_dict=torch.load('./data/parameter/transzero_origin.pkl')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#
# model_dict.update(pretrained_dict)
#
# model.load_state_dict(model_dict)

optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.9)
# main loop
niters = dataloader.ntrain * config.epochs//config.batch_size
report_interval = niters//config.epochs
best_performance = [0, 0, 0, 0]
best_performance_zsl = 0

for i in range(0, niters):
    model.train()
    optimizer.zero_grad()
    
    batch_label, batch_feature, batch_att = dataloader.next_batch(config.batch_size)
    out_package = model(batch_feature,batch_label,is_training=True)
    
    in_package = out_package
    in_package['batch_label'] = batch_label


    out_package = model.compute_loss(in_package)
    loss, loss_CE, loss_cal, loss_reg, loss_kl, loss_kl_class = out_package['loss'], out_package[
        'loss_CE'], out_package['loss_cal'], out_package['loss_reg'], out_package['loss_kl'], out_package['loss_kl_class']

    loss.backward()
    optimizer.step()

    # report result
    if i % report_interval == 0:
        print('-' * 30)
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            dataloader, model, config.device, bias_seen=0, bias_unseen=0)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs

        print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, '
              'loss_reg=%.3f, loss_kl=%.3f, loss_kl_class=%.3f| acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | '
              'acc_zs=%.3f' % (
                  i, int(i // report_interval),
                  loss.item(), loss_CE.item(), loss_cal.item(),
                  loss_reg.item(), loss_kl.item(),loss_kl_class.item(),
                  best_performance[0], best_performance[1],
                  best_performance[2], best_performance_zsl))

        wandb.log({
            'iter': i,
            'loss': loss.item(),
            'loss_CE': loss_CE.item(),
            'loss_cal': loss_cal.item(),
            'loss_reg': loss_reg.item(),
            'loss_kl': loss_kl.item(),
            'acc_unseen': acc_novel,
            'acc_seen': acc_seen,
            'H': H,
            'acc_zs': acc_zs,
            'best_acc_unseen': best_performance[0],
            'best_acc_seen': best_performance[1],
            'best_H': best_performance[2],
            'best_acc_zs': best_performance_zsl
        })

# torch.save(model.state_dict(),"./data/parameter/model1.pkl")
