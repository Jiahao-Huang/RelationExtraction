import os
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from metrics import PRMetric
from torch import optim
from models import transformer, bert, BasicModule
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def train(epoch, model, dataloader, optimizer, criterion, device, cfg):
    model.train()

    metric = PRMetric()
    losses = []

    logger.info(model.predict_type +" training:")

    for batch_idx, x in enumerate(dataloader, 1):
        predict, truth = {}, {}

        for key, value in x.items():
            x[key] = value.to(device)
        
        output = model(x)
        if model.predict_type == "rel":
            predict = output
            truth = x["relation"]
            for t in truth:
                assert t < predict.shape[-1]
        elif model.predict_type == "tail":
            predict = output.reshape(-1, cfg.vocab_size)
            truth = x['tail_cont'].reshape(-1)
            for t in truth:
                assert t < predict.shape[-1]

        loss = criterion(predict, truth)

        loss.backward()
        optimizer.step()

        metric.update(y_true=truth, y_pred=predict)
        losses.append(loss.item())

        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * x['relation'].shape[0]
        if (cfg.train_log and batch_idx % cfg.log_interval == 0) or batch_idx == len(dataloader):
            # p r f1 皆为 macro，因为micro时三者相同，定义为acc
            acc, p, r, f1 = metric.compute()
            logger.info(f'Train Epoch {epoch}: [{data_cal}/{data_total} ({100. * data_cal / data_total:.0f}%)]\t'
                        f'Loss: {loss.item():.6f}')
            logger.info(f'Train Epoch {epoch}: Acc: {100. * acc:.2f}%\t'
                        f'macro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    return losses[-1]


def validate(epoch, model, dataloader, criterion, device, cfg):
    model.eval()

    metric = PRMetric()
    losses= []

    for batch_idx, x in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)

        with torch.no_grad():
            output = model(x)

            if model.predict_type == "rel":
                predict = output
                truth = x["relation"]
                for t in truth:
                    assert t < predict.shape[-1]
            elif model.predict_type == "tail":
                predict = output.reshape(-1, cfg.vocab_size)
                truth = x['tail_cont'].reshape(-1)
                for t in truth:
                    assert t < predict.shape[-1]
            
            loss = criterion(predict, truth)

            metric.update(y_true=truth, y_pred=predict)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)

    acc, p, r, f1 = metric.compute()
    data_total = len(dataloader.dataset)

    if epoch >= 0:
        logger.info(f'Valid Epoch {epoch}: [{data_total}/{data_total}](100%)\t {model.predict_type} Loss: {loss:.6f}')
        logger.info(f'Valid Epoch {epoch}: {model.predict_type} Acc: {100. * acc:.2f}%\t')
    else:
        logger.info(f'Test Data: [{data_total}/{data_total}](100%)\t {model.predict_type} Loss: {loss:.6f}\t')
        logger.info(f'Test Data: {model.predict_type} Acc: {100. * acc:.2f}%\t')

    return f1, loss

def TRAIN(predict_type, train_dataloader, valid_dataloader, test_dataloader, device, cfg):
    logger.info('=' * 10 + f' Start {predict_type} training ' + '=' * 10)

    best_f1, best_epoch = -1, 0

     #initialize
    if (predict_type == "rel"):
        model = bert(cfg.model_rel, "rel")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.model_rel.lr, weight_decay=cfg.model_rel.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.model_rel.lr_factor, patience=cfg.model_rel.lr_patience)
        criterion = nn.CrossEntropyLoss()
    elif (predict_type == "tail"):
        model = transformer(cfg.model_tail, "tail")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.model_tail.lr, weight_decay=cfg.model_tail.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.model_tail.lr_factor, patience=cfg.model_tail.lr_patience)
        criterion = nn.CrossEntropyLoss()
    
    train_losses, valid_losses = [], []
    for epoch in range(1, cfg.epoch + 1):
        train_loss = train(epoch, model, train_dataloader, optimizer, criterion, device, cfg)
        valid_f1, valid_loss = validate(epoch, model, valid_dataloader, criterion, device, cfg)
        scheduler.step(valid_loss)
        model_path = model.save(epoch, cfg)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if best_f1 < valid_f1:
            best_f1 = valid_f1
            best_epoch = epoch
        
    logger.info(f'total {cfg.epoch} epochs, best epoch of {predict_type}: {best_epoch}')
    logger.info('=====end of training====')
    logger.info('=====start test performance====')
    validate(-1, model, test_dataloader, criterion, device, cfg)
    logger.info('=====ending====')

def TEST(test_dataloader, device, cfg):
    rel_model_path = os.path.join(cfg.cwd, cfg.trained_model, "rel_model.pth")
    tail_model_path = os.path.join(cfg.cwd, cfg.trained_model, "tail_model.pth")

    rel_model, tail_model = bert(cfg.model_rel, "rel").to(device), transformer(cfg.model_tail, "tail").to(device)
    rel_model.load_state_dict(torch.load(rel_model_path))
    tail_model.load_state_dict(torch.load(tail_model_path))

    rel_model.eval()
    tail_model.eval()

    rel_metric, tail_metric = PRMetric(), PRMetric()
    rel_losses, tail_losses = [], []

    criterion = nn.CrossEntropyLoss()
    words, relations_pred, tails_pred, relations_truth, tails_truth = [], [], [], [], []

    for batch_idx, x in enumerate(test_dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)

        with torch.no_grad():
            rel_truth = x["relation"]
            tail_truth = x['tail_cont'].reshape(-1)

            rel_predict = rel_model(x)
            tail_predict = tail_model(x, torch.argmax(rel_predict, dim=1)).reshape(-1, cfg.vocab_size)

            rel_loss = criterion(rel_predict, rel_truth)
            tail_loss = criterion(tail_predict, tail_truth)

            rel_metric.update(y_true=rel_truth, y_pred=rel_predict)
            tail_metric.update(y_true=tail_truth, y_pred=tail_predict)

            rel_losses.append(rel_loss.item())
            tail_losses.append(tail_loss.item())

            words.append(x["word"].cpu().numpy().tolist())
            relations_truth.append(x["relation"].cpu().numpy().tolist())
            tails_truth.append(x["tail_cont"].cpu().numpy().tolist())
            relations_pred.append(torch.argmax(rel_predict, dim=-1).cpu().numpy().tolist())
            tails_pred.append(torch.argmax(tail_predict, dim=-1).reshape(x["tail_cont"].shape[0], -1).cpu().numpy().tolist())

    rel_loss = sum(rel_losses) / len(rel_losses)
    tail_loss = sum(tail_losses) / len(tail_losses)

    rel_acc, _, _, _ = rel_metric.compute()
    tail_acc, _, _, _ = tail_metric.compute()
    data_total = len(test_dataloader.dataset)

    logger.info(f'Test Data: [{data_total}/{data_total}](100%)\t Relation Loss: {rel_loss:.6f}\t Tail Loss: {tail_loss:.6f}')
    logger.info(f'Test Data: Relation Acc: {100. * rel_acc:.2f}%\t Tail Acc: {100. * tail_acc:.2f}%')

    def _valid_id(id):
        return not id in [0, 101]

    def _decode(sentence, tokenizer):
        sentence = list(filter(_valid_id, sentence))
        decoded = tokenizer.decode(sentence)
        return decoded

    result_f = open(os.path.join(cfg.cwd, "result.txt"), "w")
    vocab = os.path.join(cfg.cwd, cfg.tokenizer_path, 'vocab.txt')
    tokenizer = BertTokenizer.from_pretrained(vocab)
    relation_f = open(os.path.join(cfg.cwd, cfg.data_path, 'relation.csv'))
    relation_ls = relation_f.readlines()

    for i, batch in enumerate(words):
        for j, sentence in enumerate(batch):
            relation_pred = relation_ls[relations_pred[i][j] + 1].split(',')[2]
            relation_truth = relation_ls[relations_truth[i][j] + 1].split(',')[2]
            sent = "".join(_decode(sentence, tokenizer))
            tail_pred = "".join(_decode(tails_pred[i][j], tokenizer))
            tail_truth = "".join(_decode(tails_truth[i][j], tokenizer))
            result_f.write(f"Sentence: {sent}\nTrue Relation: {relation_truth}, Predicted Relation: {relation_pred}\nTrue Tail: {tail_truth}\nPredicted Relation: {tail_pred}\n\n")
            


