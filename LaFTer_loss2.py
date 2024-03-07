import argparse
import torch
import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
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
import datasets.isic2018
import datasets.pneumonia_guangzhou
import datasets.shenzhen_cxr
import datasets.montgomery_cxr
import datasets.idrid 
import trainers.LaFTer as lafter_uft
from utils.utils import *
import os
import json
import clip
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor

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

def embedding_similarity(args,cfg, model, teloader, label_mapping):
    num_samples = 0
    alignment = []
    
    for i, (inputs) in enumerate(tqdm(teloader)):
        img = inputs["img"]
        labels = inputs["label"]
        num_samples += img.shape[0]
        
        # Create a new list based on the mapping
        classes = [label_mapping[int(num)] for num in labels] 
        class_name  = classes[0][0]

        path_to_file = f'./descriptions/generic/{cfg.DATASET.NAME}.json'
        with open(path_to_file) as f:
            gpt3_prompts = json.load(f)
        
        desc = [gpt3_prompts[class_name] for class_name in gpt3_prompts.keys()] # desc is a 2d list, (5,5)
        # breakpoint()
        desc_all = [item for sublist in desc for item in sublist]
        prompt_label = "a photo of a " + class_name
        prompts_labels_isic2018 =   [0]*50 + [1]*51 + [2]*52 + [3]*50 + [4]*50 + [5]*55 + [6]*52 
        prompts_labels_pneumonia =  [0]*78 + [1]*74
        prompts_labels_shenzhen =   [0]*65 + [1]*69 
        prompts_labels_montgomery = [0]*65 + [1]*69 
        prompts_labels_idrid      = [0]*110 + [1]*110 + [2]*110 + [3]*110 + [4]*110 
        prompts_labels = torch.tensor(prompts_labels_isic2018).cuda()
        # text_label = clip.tokenize(prompt_label)
        # text_inputs = clip.tokenize(desc_all)
        processor = model.processor
        inputs = processor(text=desc_all, return_tensors="pt", padding=True).input_ids.cuda() 
        model.cuda()
        outputs = model(inputs)
        if args.zero_shot:
            pass
        else:
            with torch.no_grad():
                img_feature = outputs['img_embeds']
                txt_feature = outputs['text_embeds']


                image_features = img_feature / img_feature.norm(dim=1, keepdim=True)
                text_features = txt_feature / txt_feature.norm(dim=1, keepdim=True)
                # cosine similarity as logits
                logit_scale = 100
                
                similarity_prompt = (logit_scale * image_features @ text_features.t())
                values, indices = similarity_prompt[0].topk(5)
                print(values)
                print(indices)

                top_k_labels = prompts_labels[indices]
                if labels in top_k_labels:
                    alignment.append(1)
                else:
                    alignment.append(0)

            alignment_score = sum(1 for item in alignment if item == 1)/len(alignment)

    return alignment_score

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


def train_txt_cls(args, model):
    optimizer, _, _ = setup_text_training_utils(args, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for i in tqdm(range(args.txt_epochs)):
        loss = model.train_txt_clas(criteria)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {i}, loss: ", loss)
    model.txt_cls_init()

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def final_entropy(model, output, optimizer, tta_steps=1, selection_p=0.1):
    selected_idx = None
    loss = 0
    for j in range(tta_steps):
        with torch.cuda.amp.autocast():
            # breakpoint()
            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, selection_p)

            loss += avg_entropy(output)

        # optimizer.zero_grad()
        # # compute gradient and do SGD step
        # loss.backward()
        # # Unscales the gradients of optimizer's assigned params in-place
        # optimizer.step()
    return loss

def add_gaussian_noise(image_tensor):
    noise = torch.randn_like(image_tensor) * args.std + args.mean
    noisy_image_tensor = image_tensor + noise
    return torch.clamp(noisy_image_tensor, 0, 1)

def train_lafter(args, model, tr_loader, val_loader):

    # first train text classifier
    train_txt_cls(args, model)

    all_acc = list()
    optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    batch_time = lossmeter()
    data_time = lossmeter()
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()
        label_list = {"label":[],"pseudo_label":[]}
        for i, batch in enumerate((tr_loader)):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            
            # add gaussian noise to the image
            # noisy_input = add_gaussian_noise(input)
            # input = noisy_input
            input = input.to(model.device)
            
            optimizer.zero_grad()
            pl = model.forward_normal_for_pl(input[0])
            out = model.forward_aug_with_prompts(input[1].float().cuda())
            
            ent_loss = final_entropy(model, out, optimizer)

            pseudo_label = F.softmax(pl, dim=-1)  # / 0.04
            arg_pl = pseudo_label.argmax(dim=1, keepdim=True) 
            arg_pl_flat = arg_pl.flatten().cuda()
            # disbale the next two lines and change loss, compare with softmax out 
            out_ = F.softmax(out, dim=-1) 
            out_ = out_.flatten().cuda()
            pseudo_label = pseudo_label.flatten().cuda()

            cr_loss = criteria(out_, pseudo_label)
            # lsce_loss = criteria(out.squeeze(), arg_pl_flat)
            # cr_loss = lsce_loss # for label smooth cross entropy
            loss = -cr_loss # for cosine similarity
            
            loss = 0.7*cr_loss + 1.5*ent_loss

            label_list["label"].append(batch["label"])
            label_list["pseudo_label"].append(arg_pl.flatten())

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

            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Evaluation: {epoch}')
        acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {acc}')
        all_acc.append(acc)
    print(f'-------------------------------- Best Accuracy: {max(all_acc)} --------------------------------')
    # save the lists
    save_path = "/home/umaima.rahman/research/sem6/LaFTer/" 
    np.savez(os.path.join(save_path,"med_acc_list_isic_cos_ent.npz"), acc_list, acc_pl_list, acc_pl_tl_list)
    # print(f'Evaluation: {args.txt_epochs}')
    # acc = test_prompting(val_loader, model)
    # print(f'TOP-1 Accuracy: {acc}')

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
    label_mapping = label_mapping_isic2018

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args
    test_loader = trainer.test_loader
    train_loader = trainer.train_loader_x

    if args.zero_shot:
        zero_shot(model, test_loader)
    else:
        train_lafter(args, model,train_loader, test_loader)
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--logfolder', default='logs', type=str)
    parser.add_argument('--lossfn', type=str)
    args = parser.parse_args()
    args.mile_stones = None
    main(args)

