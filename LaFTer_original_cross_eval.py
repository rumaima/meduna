
import argparse
import json
import torch
import datetime
from dassl.utils import setup_logger, setup_loguru, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
# custom
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.food101
import datasets.sun397
import datasets.ucf101
import datasets.imagenet_r
import datasets.imagenet
import datasets.imagenet_s
import datasets.imagenet_a
import datasets.caltech101
import datasets.cifar
import datasets.idrid 
import datasets.isic2018
import datasets.pneumonia_guangzhou
import datasets.shenzhen_cxr
import datasets.montgomery_cxr
import datasets.covid
import trainers.LaFTer as lafter_uft
from utils.utils import *
import os
import clip
from loguru import logger
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.logspec:
        cfg.LOGSPEC = args.logspec


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.txt_cls = args.txt_cls
    cfg.gpt_prompts = args.gpt_prompts


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg


class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count

def zero_shot_lafter(model, loader, pickle_z=None):
    print('-------------- ZERO SHOT INFERENCE --------------')
    total = 0.
    correct_base = 0.
    model.eval()

    if pickle_z is not None:
        N = len(loader.dataset)

        pickle_dict = {'Z' :np.zeros((N,512)),
                        'Ytrue' :np.zeros((N,)), 
                        'Yhat' :np.zeros((N,)),
                        'count': 0}
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            target = inputs['label']
            images = inputs['img']
            if isinstance(images, list):
                images = images[0]

            images = images.cuda()
            target = target.cuda()
            out = model(images)
            logits_base = out
            pred_base = torch.argmax(logits_base, dim=1)
            
            if pickle_z is not None:
                pickle_dict['Yhat'][pickle_dict['count']: pickle_dict['count']+len(images)] = pred_base.cpu()
                pickle_dict['Ytrue'][pickle_dict['count']: pickle_dict['count']+len(images)] = target.cpu()
                pickle_dict['count'] += len(images)

            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")

    if pickle_z is not None:
        # calculate F1 score, use pickle_dict['Yhat'] and pickle_dict['Ytrue']
        pickle_dict['f1'] = f1_score(pickle_dict['Ytrue'], pickle_dict['Yhat'])

        # Calculate ROC curve, use pickle_dict['Yhat'] and pickle_dict['Ytrue']
        fpr, tpr, thresholds = roc_curve(pickle_dict['Ytrue'], pickle_dict['Yhat'])
        roc_auc = auc(fpr, tpr)

        pickle_dict['fpr'] = fpr 
        pickle_dict['tpr'] = tpr 
        pickle_dict['thresh'] = thresholds
        pickle_dict['roc_auc'] = roc_auc 

        # pickle the dictionary here
        with open(pickle_z, 'wb') as f:
            pickle.dump(pickle_dict, f)
    print(f"Pickle file {pickle_z} saved")
    

def evaluate_other_datasets_lafter(dataloader, model, pickle_z=None):
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict["model"])
    model = model.cuda()
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses= []
    if pickle_z is not None:
        N = len(dataloader.dataset)

        pickle_dict = {'Z' :np.zeros((N,512)),
                        'Ytrue' :np.zeros((N,)), 
                        'Yhat' :np.zeros((N,)),
                        'count': 0}
        

    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()

    # accuracy
    for i, inputs in enumerate(tqdm(dataloader)):
        labels = inputs['label']
        inputs = inputs['img']
        if isinstance(inputs, list):
            inputs = inputs[0]
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            Z_outputs, Y_outputs = model.eval_clip_with_embeddings(inputs)
            _, predicted = Y_outputs.max(1)
            Y_pl = F.softmax(Y_outputs)
            Y_arg = Y_pl.argmax(axis=1)
            _, predicted = Y_pl.max(1)
            losses.append(criterion(Y_pl, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
            
            if pickle_z is not None:
                pickle_dict['Z'][pickle_dict['count']: pickle_dict['count']+len(inputs)] = np.array(Z_outputs.cpu())
                pickle_dict['Yhat'][pickle_dict['count']: pickle_dict['count']+len(inputs)] = Y_arg.cpu()
                pickle_dict['Ytrue'][pickle_dict['count']: pickle_dict['count']+len(inputs)] = labels.cpu()
                pickle_dict['count'] += len(inputs)

        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))

        batch_time.update(time.time() - end)
        
        end = time.time()
    model.eval()

    if pickle_z is not None:
        # calculate F1 score, use pickle_dict['Yhat'] and pickle_dict['Ytrue']
        pickle_dict['f1'] = f1_score(pickle_dict['Ytrue'], pickle_dict['Yhat'])

        # Calculate ROC curve, use pickle_dict['Yhat'] and pickle_dict['Ytrue']
        fpr, tpr, thresholds = roc_curve(pickle_dict['Ytrue'], pickle_dict['Yhat'])
        roc_auc = auc(fpr, tpr)

        pickle_dict['fpr'] = fpr 
        pickle_dict['tpr'] = tpr 
        pickle_dict['thresh'] = thresholds
        pickle_dict['roc_auc'] = roc_auc 

        # pickle the dictionary here
        with open(pickle_z, 'wb') as f:
            pickle.dump(pickle_dict, f)

    return top1.avg * 100

def dumb_max(dataloader):
    N = len(dataloader.dataset)
    all_ones = np.ones((N,))
    all_zeros = np.zeros((N,))
    all_labels = []
    for i, inputs in enumerate(tqdm(dataloader)):
        labels = inputs['label']
        all_labels.append(labels)
    complete_labels = []
    for tensor in all_labels:
        complete_labels.extend(tensor.tolist())
    f1_ones = f1_score(complete_labels, all_ones)
    f1_zeros = f1_score(complete_labels, all_zeros)
    max_f1_score = max(f1_ones, f1_zeros)
    return max_f1_score

def main(args):
    cfg = setup_cfg(args)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = cfg.DATASET.NAME
    setup_txt_epochs(args, dataset_name)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    setup_loguru(cfg.OUTPUT_DIR, cfg.LOGSPEC)
    print_args(args, cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args
    val_loader = trainer.val_loader
    test_loader = trainer.test_loader
    train_loader = trainer.train_loader_x

   

    if args.zero_shot:
        # zero_shot(model, test_loader)
        zero_shot_lafter(model, test_loader, pickle_z=args.pickle_file_path)
    else: 
        model_path = args.model_path

        print(f'Dataset:{dataset_name}')
        other_train_acc = evaluate_other_datasets_lafter(train_loader, model)
        other_val_acc = evaluate_other_datasets_lafter(val_loader, model)
        other_test_acc = evaluate_other_datasets_lafter(test_loader, model, pickle_z=args.pickle_file_path)

        len_train_loader = len(train_loader.dataset)
        len_val_loader = len(val_loader.dataset)
        len_test_loader = len(test_loader.dataset)
        total_len_data = len_train_loader + len_val_loader + len_test_loader
        print('-------------------TOP-1 Accuracy----------------------')
        print(f'TOP-1 train Accuracy: {other_train_acc}')
        print(f'TOP-1 val Accuracy: {other_val_acc}')
        print(f'TOP-1 test Accuracy: {other_test_acc}')
        logger.info(f"Evaluation on \t{dataset_name}:\t\t{other_train_acc}\t{other_val_acc}\t{other_test_acc}")

        print('--------------------------------------------------------')
        weighted_accuracy = len_train_loader * other_train_acc + len_val_loader * other_val_acc + len_test_loader * other_test_acc 
        weighted_accuracy = weighted_accuracy/total_len_data
        print(f'Weighted accuracy for dataset: {dataset_name} is {weighted_accuracy}')

    # evaluate_dumb(test_loader, train_loader, val_loader)
        # dumb_f1_score = dumb_max(test_loader)
        # print("Dataset:", dataset_name)
        # print("F1_score:", dumb_f1_score)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=7777, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='tap', required=True, choices=['cls_only',
                                                                                      'templates_only', 'lafter', 'zero_shot'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--prompt_epochs', type=int, default=100)
    parser.add_argument('--logfolder', default='logs', type=str)
    parser.add_argument('--logspec', default='logs', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--pickle_file_path', default='', type=str)
    args = parser.parse_args()
    args.mile_stones = None
    main(args)

