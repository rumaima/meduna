import argparse
import json
import torch
import datetime
from dassl.utils import setup_logger, setup_loguru, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
from trainers.mlhc_mlp import *
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
from trainers.mlhc_mlp import *
from utils.utils import *
import os
import clip
from loguru import logger

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




def test(args, teloader, model):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pl = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    one_hot_pl = []

    for i, (inputs) in enumerate(tqdm(teloader)):
        img = inputs["img"]
        labels = inputs["label"]

        if args.zero_shot:
            with torch.no_grad():
                output_pseudo_label = model(inputs.cuda(), zero_shot=True)
                _, predicted_pl = output_pseudo_label.max(1)
                one_hot_pl.append(predicted_pl.eq(labels.cuda()).cpu())
                acc1_pl = one_hot_pl[-1].sum().item() / len(labels)
                top1_pl.update(acc1_pl, len(labels))

        else:
            with torch.no_grad():
                inputs, labels = img.cuda(), labels.cuda()
                outputs = model(inputs, clip_eval=True)
                _, predicted = outputs.max(1)
                one_hot.append(predicted.eq(labels).cpu())
                acc1 = one_hot[-1].sum().item() / len(labels)
                top1.update(acc1, len(labels))

    if not args.zero_shot:
        return top1.avg * 100, top1_pl.avg * 100
    else:
        return top1_pl.avg * 100

def calculate_cluster_heads(tr_loader, model, model_t):
    model.eval()
    model_t.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses= []
    end = time.time()

    batch = next(iter(tr_loader)) # gets first element of the dataloader
    input_img = batch["img"]
    
    input_img = torch.stack(input_img)  # two views from dataloader
    input_img = input_img.to(model.device)
    
    input_image_batch = input_img[0]
    input_label_batch = batch["label"]

    m = len(input_image_batch)

    num_classes = 2


    E = model.forward_normal_no_prompts(input_image_batch)
    Z = model_t(E)

    z_0_cluster_head = Z[input_label_batch==0].mean(axis=0)
    z_1_cluster_head = Z[input_label_batch==1].mean(axis=0)

    Y = model.classifier(Z)
    Y = F.softmax(Y)

    Y_true = nn.functional.one_hot(input_label_batch, 2).cuda()
    Y_true = Y_true.to(torch.float32)

    comb = int(0.5*m*(m-1))
    loss_comp = torch.zeros((comb,1))
    vk = 0
    
    for vi in range(m):
        zi = Z[vi]
        yhat_i = Y[vi]
        yi = Y_true[vi]
        for vj in range(vi+1,m):
            zj = Z[vj]
            yhat_j = Y[vj]
            yj = Y_true[vj]
            # norm_y = torch.linalg.norm(yi-yj)
            norm_z = torch.log(torch.linalg.norm(zi-zj) + 1e-6)
            Ismall = ((yi.T@yj)*norm_z).to(torch.float32).cuda()
            Ilarge = ((1-yi.T@yj)*norm_z).to(torch.float32).cuda()
            clus_dist =  Ismall - Ilarge
            ent = -args.lambda1*(yi.T@torch.log(yhat_i) + yj.T@torch.log(yhat_j))
            loss_comp[vk] =  clus_dist + ent
            vk += 1

    clus_dist_loss =  loss_comp.mean()

    return clus_dist_loss, z_0_cluster_head, z_1_cluster_head

        
def train_mlhc(args, model, tr_loader, val_loader, test_loader, dataset_name):
    all_acc_test = list()
    all_acc_train = list()
    all_acc_val = list()

    # Create the MLP g
    mlp_input_size = 512
    mlp_hidden_size1 = 256
    mlp_hidden_size2 = 16
    mlp_output_size = 2
    model_g = BinaryClassificationMLP(mlp_input_size, mlp_hidden_size1, mlp_hidden_size2, mlp_output_size)
    model_g = model_g.to(model.device)

    # Create a transformer model
    input_size = 512  # Input embedding size
    hidden_size = 256
    num_attention_heads = 8
    num_hidden_layers = 8
    output_size = 16  # Output size for binary classification

    # Create the model
    model_t = TransformerClassifier(input_size, hidden_size, num_attention_heads, num_hidden_layers, output_size)
    model_t = model_t.to(model.device)
     

    optimizer, scheduler, criteria = setup_lafter_training_utils(args, model, model_t, model_g)
    batch_time = lossmeter()
    data_time = lossmeter()
    
    best_acc = 0.0

    # print(f'-------------------Accuracies before training-----------------------')
    # train_acc = evaluation_train_one_batch(tr_loader,model, model_t)
    # val_acc = evaluation_no_prompts(val_loader,model, model_t)
    # test_acc = evaluation_no_prompts(test_loader,model, model_t)
    
    # train_acc = np.round(train_acc,2)
    # val_acc = np.round(val_acc,2)
    # test_acc = np.round(test_acc,2)
    # print(f'TOP-1 train Accuracy: {train_acc}')
    # print(f'TOP-1 val Accuracy: {val_acc}')
    # print(f'TOP-1 test Accuracy: {test_acc}')
    # logger.info(f"\t -1\t{train_acc}\t{val_acc}\t{test_acc}")


    model_save_path = "/l/users/umaima.rahman/research/sem6/lafter_checkpoints"

    M = torch.Tensor([1, -1]).cuda()
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.classifier.train()
        
        end = time.time()

        

        for i, batch in enumerate((tr_loader)):
            
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input_img = batch["img"]
        
            input_img = torch.stack(input_img)  # two views from dataloader
            input_img = input_img.to(model.device)
            
            input_image_batch = input_img[0]
            input_label_batch = batch["label"]

            m = len(input_image_batch)
        
            num_classes = 2

            E = model.forward_normal_no_prompts(input_image_batch)
            Z = model_t(E)

            Y = model.classifier(Z)
            Y = F.softmax(Y)
            
            loss_0, z_0_cluster_head, z_1_cluster_head = calculate_cluster_heads(tr_loader, model, model_t)

            L_0 = torch.linalg.norm(Z-z_0_cluster_head, axis=1) 
            L_1 = torch.linalg.norm(Z-z_1_cluster_head, axis=1)
            # L_diff = L_0 - L_1
            L_diff = torch.log(L_0+1e-6)-torch.log(L_1+1e-6)

            Y_hat = nn.functional.one_hot(Y.argmax(1), 2).cuda()
            Y_hat = Y_hat.to(torch.float32)

            indicator = M@Y_hat.T
            
            loss = 1e1*loss_0 + (L_diff.T@indicator)/m
            
            # optimizer.zero_grad()
                
            if i % args.print_freq == 0:
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {losses}\t"
                    "lr {lr:.6e}".format(
                        epoch + 1,
                        args.epochs,
                        i + 1,
                        len(tr_loader),
                        losses=loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    ))
            # breakpoint()
            # loss.requires_grad = True
        
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        # Evaluation on validation set for model selection
        print(f'Evaluating on the val set: {epoch}')
        # train_acc = evaluation_train_one_batch(tr_loader,model, model_t) only to check one batch for train
        train_acc = evaluation_no_prompts(tr_loader,model, model_t)
        val_acc = evaluation_no_prompts(val_loader,model, model_t)
        test_acc = evaluation_no_prompts(test_loader,model, model_t)
    
        train_acc = np.round(train_acc,2)
        val_acc = np.round(val_acc,2)
        test_acc = np.round(test_acc,2)
    

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "model": model.state_dict(),
                "model_t":model_t.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            print(f'Dataset:{dataset_name}')
            print(f'Saving model for epoch: {epoch}')
            print(f'TOP-1 validation Accuracy: {val_acc}')

            torch.save(checkpoint, os.path.join(model_save_path, f"mlhc_clusdist_semi_model_best_{args.logspec}_{dataset_name}.pth"))
            
        print(f'Dataset:{dataset_name}')
        print(f'TOP-1 train Accuracy: {train_acc}')
        print(f'TOP-1 val Accuracy: {val_acc}')
        print(f'TOP-1 test Accuracy: {test_acc}')
        all_acc_test.append(test_acc)
        all_acc_val.append(val_acc)
        all_acc_train.append(train_acc)
        logger.info(f"\t {epoch}\t{train_acc}\t{val_acc}\t{test_acc}")
    print(f'-------------------------------- Best Test Accuracy: {max(all_acc_test)} --------------------------------')

# def test_mlhc(args, model, tr_loader, test_loader, dataset_name, model_clip, model_g, model_t):
#     # Now evaluate using the test evaluation
#     print(f'Evaluating on the test set:')
#     test_acc, train_acc = evaluation_no_prompts(tr_loader, val_loader, test_loader, model, model_t)
#     test_acc = np.round(test_acc,2)
#     train_acc = np.round(train_acc,2)
#     print(f'Dataset:{dataset_name}')
#     print(f'TOP-1 test Accuracy: {test_acc}')
#     print(f'TOP-1 train Accuracy: {train_acc}')

def main(args):
    cfg = setup_cfg(args)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = cfg.DATASET.NAME
    setup_txt_epochs(args, dataset_name)

    label_mapping_isic2018 = {
            0: ['MEL', 0, 50],
            1: ['DF',51, 100],
            2: ['BCC',101, 152],
            3: ['BKL',153, 202],
            4: ['AKIEC',203, 252],
            5: ['NV',253, 307],
            6: ['VASC',308, 359]
        }
    label_mapping_pneumonia = {
            0: ['NORMAL', 0, 77],
            1: ['PNEUMONIA',78, 151],
        }
    label_mapping_shenzhen = {
            0: ['NORMAL', 0, 64],
            1: ['TB',65, 133],
        }
    label_mapping_montgomery = {
            0: ['NORMAL', 0, 64],
            1: ['TB',65, 133],
        }
    label_mapping_idrid = {
            0: ['Normal', 0, 109],
            1: ['Stage_1_Retinopathy',110, 219],
            2: ['Stage_2_Retinopathy',220, 329],
            3: ['Stage_3_Retinopathy',330, 439],
            4: ['Stage_4_Retinopathy',440, 549]
        }
        # to be changed according to the dataset
    label_mapping = label_mapping_pneumonia


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
        zero_shot(model, test_loader)
    else:
        print(f'Dataset:{dataset_name}')
        train_mlhc(args, model, train_loader, val_loader, test_loader, dataset_name)
        # test_mlhc(args, model, train_loader, test_loader, dataset_name, model_clip, model_g, model_t)
        # train_lafter(args, model,train_loader, test_loader)
        # alignment_score = embedding_similarity(args, cfg, model, test_loader, label_mapping)
        # print(f"\nAlignment score:\n {alignment_score:.2f}")

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
    args = parser.parse_args()
    args.mile_stones = None
    main(args)

