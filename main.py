import os
import sys
import argparse
from datetime import datetime


import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm, trange
# from accelerate import Accelerator
import transformers
from sklearn.metrics import f1_score
import wandb


from model import CLS_model
from process import Processor
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=str, default="True")
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, required=False)
    parser.add_argument("--eps", type=float, default=1e-8, required=False, help="AdamW中的eps")
    parser.add_argument("--seed", type=int, default=20220924)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--pretrained_path", type=str, default="./bart-base-chinese")
    parser.add_argument("--train_path", type=str, default="../Data/Dataset/CCL2018_data_3_train.json")
    parser.add_argument("--dev_path", type=str, default="../Data/Dataset/CCL2018_data_3_valid.json")
    parser.add_argument("--test_path", type=str, default="../Data/Dataset/CCL2018_data_3_valid.json", required=False)
    parser.add_argument("--output", type=str, default="./saved_models")
    parser.add_argument("--label", type=str, default="exp2")
    parser.add_argument("--train_num", type=int, default=-1, required=False)
    parser.add_argument("--dev_num", type=int, default=-1, required=False)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--warmup_steps', type=int, default=600, help='warm up steps')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument("--is_resume", type=str, default="False", help="是否重新恢复训练")
    parser.add_argument("--resume_checkpoint_path", type=str, default="", required=False, help="训练断点文件的路径")



    args = parser.parse_args()
    # output = os.path.join(args.output, args.label)
    # utils.create_dir(output)
    # logger = utils.Logger(output + "/args.txt")
    # for arg in vars(args):
    #     logger.write("%s: %s" % (arg, getattr(args, arg)))
    return args

def load_data(logger, config):
    """
    load train_dataset and dev_dataset

    Return (train_dataloader, dev_dataloader)
    """
    logger.info("loading training dataset and validating dataset!")
    processor = Processor(config=config)
    train_data, train_labels = processor.get_data(mode="train")
    dev_data, dev_labels = processor.get_data(mode="dev")

    if config.train_num > 0:
        train_data = train_data[:config.train_num]
    if config.dev_num > 0:
        dev_data = dev_data[:config.dev_num]
    logger.info("train_length: %d" % len(train_data))
    logger.info("valid_length: %d" % len(dev_data))

    train_loader = processor.create_dataloader(
        train_data, train_labels, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = processor.create_dataloader(
        dev_data, dev_labels, batch_size=config.batch_size, shuffle=False
    )

    return train_loader, valid_loader

def load_test_data(logger, config, mode="same"):
    """
    load test_dataset

    Return test_dataloader, tokenizer
    """
    logger.info("loading test dataset!")
    if mode == "same":
        processor = Processor(config = config)
        test_data, test_labels = processor.get_data(mode="test")
        logger.info("test_length: %d" % len(test_data))

        test_loader = processor.create_dataloader(
            test_data, test_labels, batch_size=config.batch_size, shuffle=False
        )
    else:
        pass
    
    return test_loader, processor.tokenizer


def train_epoch(model:CLS_model, train_dataloader, optimizer, scheduler, logger, epoch, config, device):
    """
    train model one epoch
    """
    model.train()
    epoch_start_time = datetime.now()
    # 记录下整个epoch的每个batch的loss总和
    total_loss = 0
    # 记录下整个epoch的pred和label
    preds_epoch, labels_epoch = [], []
    
    for batch_idx, (input_ids, labels, attention_mask) in enumerate(tqdm(train_dataloader, desc="Training Epoch %d:" % (epoch + 1))):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)    # (batch_size, seq_len)
            labels = labels.to(device)      # (batch_size,)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            logits = outputs.logits     # (batch_size, num_labels)
            pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)     # (batch_size,)
            # 对多块显卡计算的loss取平均
            loss = loss.mean()
            
            preds_epoch.append(pred)
            labels_epoch.append(labels)

            batch_macro_f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average="macro")
            total_loss += loss.item()
            # 对loss平均
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # 进行一定step的梯度累积后，更新参数
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度
                optimizer.zero_grad()

            wandb.log({"lr": scheduler.get_last_lr()[0]})

            if (batch_idx + 1) % config.log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss: {}, batch_macro_f1: {}, lr: {}".format(
                        batch_idx + 1, epoch + 1, loss.item() * config.gradient_accumulation_steps, batch_macro_f1, scheduler.get_last_lr()
                    )
                )
            

        except RuntimeError as exc:
            if "out of memory" in str(exc):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exc))
                raise exc
    
    # 记录当前epoch的macro_f1和平均loss
    preds_epoch = torch.cat(preds_epoch, dim = 0)
    labels_epoch = torch.cat(labels_epoch, dim = 0)
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_macro_f1 = f1_score(labels_epoch.cpu().numpy(), preds_epoch.cpu().numpy(), average="macro")
    logger.info(
        "epoch {}: loss: {}, macro_f1: {}".format(epoch + 1, epoch_mean_loss, epoch_macro_f1)
    )
    wandb.log({"epoch": epoch + 1, "train_loss":epoch_mean_loss, "train_macro_f1":epoch_macro_f1})
    # save model
    # model_path = os.path.join(config.output, "epoch{}".format(epoch + 1))
    # if not os.path.exists(model_path):
    #     utils.create_dir(model_path)
    # model.save_model(model_path)
    logger.info("epoch {} finished".format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info("time for one epoch: {}".format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss

def validate_epoch(model, validate_dataloader, logger, epoch, config, device):
    logger.info("start validating")
    model.eval()
    epoch_start_time = datetime.now()
    total_loss = 0
    preds_epoch, labels_epoch = [], []
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels, attention_mask) in enumerate(tqdm(validate_dataloader, desc="Validation: ")):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                logits = outputs.logits
                loss = loss.mean()
                pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)

                preds_epoch.append(pred)
                labels_epoch.append(labels)

                total_loss += loss.item()
            
            # 记录当前epoch的平均loss
            preds_epoch = torch.cat(preds_epoch, dim = 0)
            labels_epoch = torch.cat(labels_epoch, dim = 0)
            epoch_macro_f1 = f1_score(labels_epoch.cpu().numpy(), preds_epoch.cpu().numpy(), average="macro")

            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}, macro_f1 {}".format(epoch + 1, epoch_mean_loss, epoch_macro_f1)
            )
            wandb.log({"dev_loss":epoch_mean_loss, "dev_macro_f1": epoch_macro_f1})
            epoch_finish_time = datetime.now()
            logger.info("time for validating one epoch: {}".format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss, epoch_macro_f1

    except RuntimeError as exc:
        if "out of memory" in str(exc):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exc))
            raise exc


def train(model, logger, train_dataloader, dev_dataloader, config, device):
    # 计算总共更新多少次梯度
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.epochs
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = config.warmup_steps, num_training_steps = t_total
    )

    # loading checkpoint if resume
    if eval(config.is_resume):
        checkpoint = torch.load(config.resume_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info("starting training")

    # 记录每个epoch的训练和验证loss
    train_losses, dev_losses = [], []
    # 记录验证集最高的f1
    best_dev_f1 = -1e9

    for epoch in trange(config.epochs, desc="Epoch"):
        # train
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, config=config, device=device
        )
        train_losses.append(train_loss)

        # validate
        dev_loss, dev_f1 = validate_epoch(
            model=model, validate_dataloader=dev_dataloader,
            logger=logger, epoch=epoch, config=config, device=device
        )
        dev_losses.append(dev_loss)

        # 保存当前f1最高的模型
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            logger.info("saving current best model for epoch {}".format(epoch + 1))
            model_path = os.path.join(config.output, config.label, "best_model_epoch_{}_f1_{}".format(epoch+ 1, dev_f1))
            utils.create_dir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_model(model_path)

    logger.info("training finished!")
    logger.info("train_losses:{}".format(train_losses))
    logger.info("dev_losses:{}".format(dev_losses))

def test(model, logger, test_dataloader, config, device, tokenizer):
    logger.info("starting testing!")
    model.eval()
    epoch_start_time = datetime.now()
    preds, texts = [], []
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels, attention_mask) in enumerate(test_dataloader):
                input_ids = input_ids.to(device)    # [batch_size, seq_len]
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, labels = labels, attention_mask=attention_mask)
                logits = outputs.logits     # [batch_size, num_classes]

                text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)       # [batch_size, text_len]. the text without special tokens 
                pred = torch.argmax(F.softmax(logits, dim = 1), dim = 1)        # [batch_size, ]

                preds.append(pred.tolist())
                texts.append(text) 
            
            logger.info("Test Finished!")
            # 将所有sentence集中到一个大列表中
            texts = [sentence for batch in texts for sentence in batch]
            return list(map(lambda x:"".join(x.split(" ")), texts)), preds
    except RuntimeError as exc:
        if "out of memory" in str(exc):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exc))
            raise exc
            


if __name__ == "__main__":
    # args
    config = parse_args()

    if eval(config.is_train):
        wandb.init(project="pun_detection", name=config.label)
        wandb.config = config
    
    # set random seed 
    utils.set_random_seed(config.seed)

    # set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,5,6"

    logger = utils.create_logger(config)

    # 创建模型的输出目录
    utils.create_dir(config.output)
    
    # build model
    model = CLS_model(config)
    logger.info("model config:\n{}".format(model.return_model_config()))

    if len(config.gpu) > 1 and torch.cuda.device_count() > 1:
        device = "cuda:%s" % config.gpu[0]
        model = model.to(device)
        model = DataParallel(model, device_ids=[int(i) for i in config.gpu.split(',')])
    else:
        device = "cuda:%s" % config.gpu
        model = model.to(device)
    logger.info("using device: %s" % config.gpu)

    # compute the amount of model's parameters
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info("number of model parameters: {}".format(num_parameters))

    # notes the config
    logger.info("config: {}".format(config))

    # loading the train_dataloader and dev_dataloader
    train_dataloader, dev_dataloader = load_data(logger=logger, config=config)

    # loading the test_dataloader
    test_dataloader, tokenizer = load_test_data(logger=logger, config=config, mode="same")

    if eval(config.is_train):
        wandb.watch(model, log="all")
        train(model, logger, train_dataloader, dev_dataloader, config, device)
    else:
        test(model, logger, test_dataloader, config, device, tokenizer)

