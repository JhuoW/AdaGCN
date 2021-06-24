from utils import *
from argparse import ArgumentParser
from earlystopping import EarlyStopping, stopping_args
from Base import GraphSGConvolution
from torch.utils.data import TensorDataset, DataLoader

from SAMMER import SAMMER
from logger import Logger
def get_dataloaders(all_idx: list, labels, batch_size = None):
    if batch_size is None:
        batch_size = max((val.numel() for val in all_idx))
    datasets = [TensorDataset(ind, labels[ind]) for ind in all_idx] 
    dataloaders = [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]
    return dataloaders

def train(args, model: GraphSGConvolution, aug_adj, features, loaders, optimizer,early_stopping, logger: Logger):
    ab = SAMMER(args.n_nodes, args.n_classes)
    prop = torch.clone(features).cuda()
    aug_adj = aug_adj.cuda()
    sample_weight = torch.ones(args.n_nodes).cuda()
    total = 0
    for layer in range(args.L):
        logger.add_line()
        logger.log("\t\t%d th Layer" % layer)
        logger.add_line()

        x = torch.spmm(aug_adj, prop)  # AX
        epoch_stats = {'train': {}, 'stopping': {}}
        # re-normalize sample weights
        sample_weight.div_(torch.sum(sample_weight))
        
        for epoch in range(early_stopping.max_epochs):
            model.train()
            running_loss = 0
            running_corrects = 0
            for idx, labels in loaders[0]:
                idx = idx.cuda()   # train_idx
                labels = labels.cuda()
                optimizer.zero_grad()

                out = model(x) 
                
                w_loss = model.weighted_loss(out, idx, labels, sample_weight)
                l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                loss = w_loss + args.l2_reg/ 2 * l2_reg
                # minimizing current weighted loss
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.item()
            running_corrects = val(model, loaders[1], x)
            epoch_stats['train']['loss'] = running_loss / len(loaders[0].dataset)
            epoch_stats['stopping']['acc'] = running_corrects / len(loaders[1].dataset)
            log = "Current Layer: {:03d}, Epoch: {:03d}, Loss: {:.4f}, Val_Acc: {:.4f}"
            print(log.format(layer, epoch, epoch_stats['train']['loss'],  epoch_stats['stopping']['acc']))
            if len(early_stopping.stop_vars) > 0:
                stop_vars = [epoch_stats['stopping'][key]
                            for key in early_stopping.stop_vars]
                if early_stopping.check(stop_vars, epoch):
                    break
        
        with torch.no_grad():
            model.eval()
            # Update sample weight
            sample_weight = ab.boost_real(model, x, labels,idx, sample_weight)
            
            individual_pred = ab._samme_proba(args.n_classes)
            # Compute the individual prediction
            total += individual_pred
            # Update feature matrix
            prop = x
    return total




@torch.no_grad()
def val(model, loader, prop):
    model.eval()
    running_corrects = 0
    for idx, labels in loader:
        x = model(prop)
        preds = torch.argmax(x, dim = -1)[idx]
        running_corrects += torch.sum(preds.cpu().detach() == labels)
    return running_corrects
    

def main(args, logger):
    ori_adj, aug_adj, features, labels, train_mask, val_mask, test_mask, n_nodes, nfeat, n_classes = load_data(args.dataset_str)
    args.n_nodes = n_nodes
    args.nfeat = nfeat
    args.n_classes = int(n_classes)
    model = GraphSGConvolution(nfeat, args.hidden_dim, n_classes, bias = args.bias, dropout= args.dropout).cuda()
    early_stopping = EarlyStopping(model, **stopping_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    if not args.public_split:
        """ PPNP split """
        train_idx, val_idx, test_idx = gen_splits(args, labels.numpy())
    else:
        """ 用public split做做看 """
        train_idx = train_mask.nonzero().squeeze()
        val_idx = val_mask.nonzero().squeeze()
        test_idx = test_mask.nonzero().squeeze()

    loaders = get_dataloaders([train_idx, val_idx, test_idx], labels)

    total = train(args, model, aug_adj,features, loaders, optimizer, early_stopping, logger)
    acc = accuracy(total.cpu(), labels)
    print("test acc: ", acc.item())  



        



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_str',default='cora', type = str)
    parser.add_argument('--n_mlp', type = int, default=2)   # number of mlp layers
    parser.add_argument('--patience', type = int, default= 300)
    parser.add_argument('--lr', type=float, default=.01) 
    parser.add_argument('--bias', type = bool, default= False)
    parser.add_argument('--hidden_dim', type = int, default= 5000)
    parser.add_argument('--dropout', type = float, default= .0)
    parser.add_argument('--l2_reg', type = float, default=5e-3)
    parser.add_argument('--weight_decay', type = float, default=1e-4)
    parser.add_argument('--L', type = int, default=12)   # in the paper, layers of cora is 12 
    parser.add_argument('--public_split', action='store_true', default= False)
    parser.add_argument('--n_train_per_class', type = int, default= 20)
    parser.add_argument('--n_val', type = int, default=500)
    parser.set_defaults(max_epochs=500)
    args = parser.parse_args()
    logger = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()
    main(args, logger)