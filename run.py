#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from utils import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader
from model import Att_BLSTM
from evaluate import Eval
from plots import plot_learning_curves, plot_confusion_matrix
from myUtils import evaluate

def print_result(predict_label, id2rel, start_idx=8001):
    with open('predicted_result.txt', 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


def train(model, criterion, loader, config):
    train_loader, dev_loader, _ = loader
    optimizer = optim.Adadelta(
        model.parameters(), lr=config.lr, weight_decay=config.L2_decay)

    print(model)
    print('training model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval(config)
    max_f1 = -float('inf')
    for epoch in range(1, config.epoch+1):
        for step, (data, label) in enumerate(train_loader):
            model.train()
            data = data.to(config.device)
            label = label.to(config.device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
            optimizer.step()

        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader)
        f1, dev_loss, _ = eval_tool.evaluate(model, criterion, dev_loader)

        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev (test): %.4f'
              % (epoch, train_loss, dev_loss, f1), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), os.path.join(config.model_dir, 'model.pkl'), _use_new_zipfile_serialization=False)
            print('>>> save model!')
        else:
            print()


def test(model, criterion, loader, config):
    print('--------------------------------------')
    print('start test ...')

    _, _, test_loader = loader
    model.load_state_dict(torch.load(
        os.path.join(config.model_dir, 'model.pkl')))
    eval_tool = Eval(config)
    f1, test_loss, predict_label = eval_tool.evaluate(
        model, criterion, test_loader)
    print('test_loss: %.3f | micro f1 on test:  %.4f' % (test_loss, f1))
    return predict_label


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    print('load_embedding() success');
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    print('get_relation() success');
    loader = SemEvalDataLoader(rel2id, word2id, config)
    print('SemEvalDataLoader success');

    train_loader, dev_loader = None, None
    if config.mode == 1:  # train mode
        train_loader = loader.get_train()
        dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finished loading train, dev and test')

    print('--------------------------------------')
    print('building Att_BLSTM model')
    model = Att_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    print('done building model')

    if config.mode == 1:  # train mode
        print('training the model')
        train(model, criterion, loader, config)
    print('using the model to make predictions')
    predict_label = test(model, criterion, loader, config)
    print_result(predict_label, id2rel)
    
    # Create confusion matrix
    #test_loss, test_accuracy, test_results = evaluate(model, config.device, test_loader, criterion)
    #class_names = ['present', 'absent', 'hypothetical', 'possible', 'conditional', 'associated_with_someone_else']
    #plot_confusion_matrix(test_results, class_names)
    
    
