import torch
import torch.optim as optim
import torch.nn.functional as F
from peft import PeftModel
from learn2learn.algorithms import MAML
from tqdm import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import minmax_scale
from .utils.model import pack_lora_layers, replace_modules
from .utils.score import pairwise_ranking_loss, listwise_ranking_loss, BT_loss, KLloss, get_substitution_matrix
import sys

def get_optimizer(optimizer, lr, params):
    params = filter(lambda p: p.requires_grad, params)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'nag':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    elif optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError('Unknown optimizer: ' + optimizer)

class TrainerBase(ABC):
    def __init__(self, model, optimizer='adam', lr=1e-4, epochs=100,
                 max_grad_norm=5, lr_decay=None, eval_metric='spearmanr',
                 log_metrics=['spearmanr'], save_dir=None, patience=5, overwrite=True):
        self.model = model
        self.optimizer = get_optimizer(optimizer, lr, model.parameters())
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        if lr_decay:
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
        self.eval_metric = eval_metric
        self.log_metrics = log_metrics
        self.save_dir = save_dir
        self.patience = patience
        self.overwrite = overwrite
        self.curr_epoch = self.curr_iter = self.best_epoch = 0
        self.best_score = float('-inf')
        self.logs = defaultdict(list)
    
    def save_states(self):
        print('Saving model states...')
        save_dir = self.save_dir if self.overwrite else f'{self.save_dir}/epoch_{self.curr_epoch}'
        print(save_dir)
        self.model.save_pretrained(save_dir)
        torch.save(self.logs, self.save_dir + '/logs.pkl')
    
    @abstractmethod
    def predict(self, batch, sup):
        pass
    
    @abstractmethod
    def compute_loss(self, batch):
        pass
    
    def train_step(self, batch): # perform one gradient update
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
    
    @abstractmethod
    def compute_metrics(self, predicts, targets, labels):
        ''' Return a dict of metrics'''
        pass
    
    def evaluate_epoch(self, eval_iter, is_active=False):
        self.model.eval()
        predicts, targets, labels = [], [], []
        pbar = tqdm(eval_iter, desc='Evaluating')
        if "use_msa" in dir(self):
            use_msa = self.use_msa
        else:
            use_msa = False
        if use_msa:
            sup = get_substitution_matrix(self.tokenizer,self.seq_aln_file)
        else:
            sup = None
        with torch.no_grad():
            for batch in pbar:
                batch_preds = self.predict(batch, sup)
                if is_active:
                    logits = batch_preds[1]
                #if use contrastive learning, this may output 2 
                batch_preds = batch_preds[0]
                
                # compute metrics on full data
                predicts.append(batch_preds.to('cpu'))
                targets.append(batch['targets'].to('cpu'))
                labels.append(batch['labels'].to('cpu'))
        
        predicts, targets, labels = torch.cat(predicts), torch.cat(targets), torch.cat(labels)
        logs = self.compute_metrics(predicts, targets, labels)
        for key, value in logs.items():
            print('{}: {:.3f}'.format(key, value))
        if is_active:
            return predicts, logs, logits
        else:
            return predicts, logs
    
    def train_epoch(self, train_iter):
        self.model.train()
        logs = dict(train_loss=0, lr=0)
        pbar = tqdm(train_iter, desc=f'Training epoch {self.curr_epoch + 1}')
        
        #假如是元学习的阶段，那么-tb是1，所以一个batch就是一个辅助任务，类型为元的sequencedata
        #是普通的训练则就是一个普通的train batch
        for batch in pbar:
            loss = self.train_step(batch)
            self.curr_iter += 1
            #将当前辅助任务的loss添加到logs中
            logs['train_loss'] += loss
            pbar.set_postfix(loss=loss)
        
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        #将平均loss添加到logs中
        logs['train_loss'] /= len(train_iter)
        print('train_loss: {:.3f}'.format(logs['train_loss']))
        logs['lr'] = self.optimizer.param_groups[0]['lr']
        print('lr: {:.1e}'.format(logs['lr']))
        return logs
    
    def __call__(self, train_iter, eval_iter=None):
        for epoch in range(self.epochs):
            #得到一个字典含有lr和train_loss
            logs = self.train_epoch(train_iter)
            for key, value in logs.items():
                #更新自身属性
                self.logs[key].append(value)
            self.curr_epoch += 1
            if eval_iter is None:
                continue
            #验证集上的指标比如spearman
            _, logs = self.evaluate_epoch(eval_iter)
            for key, value in logs.items():
                self.logs[key].append(value)
            
            score = logs[self.eval_metric]
            if score > self.best_score:
                self.best_epoch = self.curr_epoch
                self.best_score = score
                if self.save_dir:
                    self.save_states()
            elif self.curr_epoch - self.best_epoch >= self.patience:
                print(f'Early stopped at epoch {self.curr_epoch}')
                print(f'Best validating {self.eval_metric} reached at epoch {self.best_epoch}: {self.best_score:.3f}')
                break
            
        if self.save_dir and eval_iter is None:
            self.save_states()
        return self.logs

class ContrastiveTrainer(TrainerBase):
    def __init__(self, model, model_reg=None, tokenizer=None, score_fn=None, use_msa=False, 
                 seq_aln_file=None, sample_ratio=None, sample_times=None, alpha=None, **kwargs):
        super().__init__(model, **kwargs)
        self.model_reg = model_reg
        self.tokenizer = tokenizer
        self.score_fn = score_fn
        self.use_msa = use_msa
        self.seq_aln_file = seq_aln_file
        self.sample_ratio = sample_ratio
        self.sample_times = sample_times
        self.alpha = alpha
        
    def predict(self, batch, sup=None):
        if self.score_fn is not None:
            return self.score_fn(self.model, batch)
        
        #wt_logit = self.model(**batch['wt_seq']).logits
        logits = self.model(**batch['wt_seq']).logits
        log_probs = torch.log_softmax(logits, dim=-1) # num_sequences * length * num_aa
        predicts = []
        for inv_idx, positions, wt_aas, mt_aas in zip(
                batch['inv_seq_idx'], batch['positions'], batch['wt_aas'], batch['mt_aas']):
            #依据inv_idx存储的该批次所用record的idx取出指定的预测结果
            
            log_prob = log_probs[inv_idx]
            if self.use_msa:
                if sup is not None:
                    subs_matrix, aln_start, aln_end = sup
                    aln_modify_logits = (1-self.alpha) * log_prob[aln_start: aln_end, :] + self.alpha * subs_matrix
                    log_prob = torch.cat([log_prob[:aln_start], aln_modify_logits, log_prob[aln_end:]], dim=0)
            predict = log_prob[positions, mt_aas] - log_prob[positions, wt_aas]
            predicts.append(predict.sum().unsqueeze(0))
        return [torch.cat(predicts), logits]
        #return [torch.cat(predicts)]
    
    def compute_loss(self, batch):
        #predicts, wt_logit_reg = self.predict(batch)
        predicts = self.predict(batch)[0]
        predicts, targets = predicts[batch['inv_list_idx']], batch['targets'][batch['inv_list_idx']]
        #if 'adapt_batches' in batch.keys():
        #    #meta learning
        #    wt_seq = list(batch['adapt_batches'])[0][0]['wt_seq']
        #    wt_logit_reg = self.model_reg(**wt_seq).logits
        #else: 
        #    #meta transfer or finetune
        #    wt_logit_reg = self.model_reg(**batch['wt_seq']).logits
        l_BT = BT_loss(predicts, targets)
        #seq = batch['wt_seq']['input_ids']
        #attn_mask = batch['wt_seq']['attention_mask']
        #l_reg = KLloss(wt_logit, wt_logit_reg, seq, attn_mask)
        #l_total = l_BT + 0.1*l_reg     
        return l_BT
    
    def compute_metrics(self, predicts, targets, labels):
        logs = {}
        for metric in self.log_metrics:
            if metric == 'spearmanr':
                logs[metric] = spearmanr(predicts, targets).statistic
            elif metric == 'ndcg':
                std_tgts = minmax_scale(targets.unsqueeze(0), (0, 5), axis=1)
                logs[metric] = ndcg_score(std_tgts, predicts.unsqueeze(0), k=10)
            elif metric == 'topk_pr':
                k = min(len(predicts), 30)
                indices = predicts.topk(k).indices
                logs[metric] = torch.count_nonzero(labels[indices]).item() / k
            else:
                raise ValueError('Unknown metric: ' + metric)
        return logs
        
class MetaContrastiveTrainer(ContrastiveTrainer):
    def __init__(self, model, adapt_lr=1e-4, first_order=True, **kwargs):
        super().__init__(model, **kwargs)
        if isinstance(model, PeftModel):
            self.adapter_name, adapter = pack_lora_layers(model)
            self.adapter = MAML(adapter, adapt_lr, first_order)
        else:
            self.model = MAML(model, adapt_lr, first_order, allow_nograd=True)
    
    def fast_adapt(self, adapt_batch, eval_batch, training=True, is_active=False):
        # copy model for meta-training
        if isinstance(self.model, PeftModel): # replace the adapter with a cloned one
            cloned_adapter = self.adapter.clone()
            replace_modules(self.model, self.adapter_name, cloned_adapter.module)
            adapt = cloned_adapter.adapt
        else: # simply copy the full model
            backup = self.model
            self.model = self.model.clone()
            adapt = self.model.adapt
        
        for batch in adapt_batch:
            adapt_loss = self.compute_loss(batch)
            adapt(adapt_loss)
        
        if training: # compute loss for training, eval_batch should be ranking dataset
            output = self.compute_loss(eval_batch)
        else: # make predictions for testing, eval_batch should be regular dataset
            with torch.no_grad():
                self.model.eval()
                output = self.predict(eval_batch)
                if not is_active:
                    output = output[0]
                self.model.train()
        
        if isinstance(self.model, PeftModel):
            replace_modules(self.model, self.adapter_name, self.adapter.module)
        else:
            self.model = backup
            
        return output
    
    def train_step(self, batch):
        loss = []
        self.optimizer.zero_grad()
        #一个step内就是一个完整的adapt_batch和一个批次的eval_batch
        for adapt_batch, eval_batch in zip(batch['adapt_batches'], batch['eval_batches']):
            eval_loss = self.fast_adapt(adapt_batch, eval_batch)
            if not eval_loss.isfinite():
                continue
            #计算梯度，并将损失添加到总的列表中
            eval_loss.backward()
            loss.append(eval_loss.item())
        if len(loss) == 0:
            return 0.
        #由于这个梯度累积了num_eval_batch次，所以需要除掉这个次数
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1. / len(loss))
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        #更新参数
        self.optimizer.step()
        return sum(loss) / len(loss)
    
    #调用fast_adapt方法不启用training模式直接得到predict的结果
    def adapt_predict(self, batch, is_active=False):
        predicts = []
        for adapt_batch, eval_batch in zip(batch['adapt_batches'], batch['eval_batches']):
            eval_preds = self.fast_adapt(adapt_batch, eval_batch, training=False, is_active=is_active)
            if is_active:
                eval_preds, logits = eval_preds[0], eval_preds[1]
            predicts.append(eval_preds)
        #长度为num_batch的列表，每个元素为batch_size,1的score
        if is_active:
            return predicts, logits
        else:
            return predicts
    
    def compute_metrics(self, predicts, targets, labels):
        logs = defaultdict(float)
        for batch_preds, batch_tgts, batch_labels in zip(predicts, targets, labels):
            log = super().compute_metrics(batch_preds, batch_tgts, batch_labels)
            for key, value in log.items():
                logs[key] += value
        
        for key in logs.keys():
            logs[key] /= len(predicts)
        return logs
    
    def evaluate_epoch(self, eval_iter, is_active=False):
        self.model.train()
        predicts = []
        logs = defaultdict(float)
        pbar = tqdm(eval_iter, desc='Evaluating')
        
        for batch in pbar:
            #得到一个列表，长度批量个数，元素为该batch下各record的score(batch_size,1)
            if is_active:
                batch_preds, logits = self.adapt_predict(batch, is_active)
            else:
                batch_preds = self.adapt_predict(batch, is_active)
            batch_preds = [preds.to('cpu') for preds in batch_preds]
            #长度为批量个数
            predicts.append(batch_preds)
            batch_tgts = [eval_batch['targets'].to('cpu') for eval_batch in batch['eval_batches']]
            batch_labels = [eval_batch['labels'].to('cpu') for eval_batch in batch['eval_batches']]
            
            log = self.compute_metrics(batch_preds, batch_tgts, batch_labels)
            pbar.set_postfix({self.eval_metric: log[self.eval_metric]})
            for key, value in log.items():
                logs[key] += value
        
        predicts = [torch.cat(preds) for preds in zip(*predicts)]
        for key in logs.keys():
            logs[key] /= len(eval_iter)
            print('{}: {:.3f}'.format(key, logs[key]))
        if is_active:
            return predicts, logs, logits
        else:
            return predicts, logs        

class RankingTrainer(TrainerBase):
    def __init__(self, model, margin=1.0, pair_fn='hinge', score_fn=None, **kwargs):
        super().__init__(model, **kwargs)
        self.margin = margin
        self.pair_fn = pair_fn
        self.score_fn = score_fn
    
    def predict(self, batch, sup=None):
        if self.score_fn is not None:
            return self.score_fn(self.model, batch)
        
        logits = self.model(**batch['sequences']).logits
        log_probs = torch.log_softmax(logits, dim=-1) # batch_size * length * num_aa
        
        predicts = []
        for inv_idx, positions, wt_aas, mt_aas in zip(
                batch['inv_seq_idx'], batch['positions'], batch['wt_aas'], batch['mt_aas']):
            #依据inv_idx存储的该批次所用record的idx取出指定的预测结果
            log_prob = log_probs[inv_idx]
            #计算fitness
            predict = log_prob[positions, mt_aas] - log_prob[positions, wt_aas]
            predicts.append(predict.sum().unsqueeze(0))
        #batch_size, 1
        return [torch.cat(predicts)]
    
    def compute_loss(self, batch):
        predicts = self.predict(batch)[0]    
        predicts, targets = predicts[batch['inv_list_idx']], batch['targets'][batch['inv_list_idx']]
        list_size = batch['inv_list_idx'].shape[1]
        if list_size == 1:
            loss = F.mse_loss(predicts, targets)
        elif list_size == 2:
            loss = pairwise_ranking_loss(predicts[:,0], predicts[:,1], targets[:,0], targets[:,1],
                                         self.pair_fn, self.margin)
        else:
            loss = listwise_ranking_loss(predicts, targets)      
        return loss
    
    def compute_metrics(self, predicts, targets, labels):
        logs = {}
        for metric in self.log_metrics:
            if metric == 'spearmanr':
                logs[metric] = spearmanr(predicts, targets).statistic
            elif metric == 'ndcg':
                std_tgts = minmax_scale(targets.unsqueeze(0), (0, 5), axis=1)
                logs[metric] = ndcg_score(std_tgts, predicts.unsqueeze(0), k=10)
            elif metric == 'topk_pr':
                k = min(len(predicts), 30)
                indices = predicts.topk(k).indices
                logs[metric] = torch.count_nonzero(labels[indices]).item() / k
            else:
                raise ValueError('Unknown metric: ' + metric)
        return logs
    
class MetaRankingTrainer(RankingTrainer):
    def __init__(self, model, adapt_lr=1e-4, first_order=True, **kwargs):
        super().__init__(model, **kwargs)
        if isinstance(model, PeftModel):
            self.adapter_name, adapter = pack_lora_layers(model)
            self.adapter = MAML(adapter, adapt_lr, first_order)
        else:
            self.model = MAML(model, adapt_lr, first_order, allow_nograd=True)
    
    def fast_adapt(self, adapt_batch, eval_batch, training=True):
        # copy model for meta-training
        if isinstance(self.model, PeftModel): # replace the adapter with a cloned one
            #克隆适配器，然后将模型中的适配器替换为克隆的适配器。避免直接修改原始适配器，确保每个任务都能独立更新自己的适配器。
            cloned_adapter = self.adapter.clone()
            replace_modules(self.model, self.adapter_name, cloned_adapter.module)
            adapt = cloned_adapter.adapt
        else: # simply copy the full model
            #如果模型是普通模型，则直接复制整个模型（通过 self.model.clone()）来确保任务间的独立性
            backup = self.model
            self.model = self.model.clone()
            adapt = self.model.adapt
        #此处为内循环
        #adapt_batch是一个列表，每个元素是一个adapt_batch的batch的，大小由-mtb决定，这个列表的长度由-as决定
        for batch in adapt_batch:
            #计算当前损失，更新模型
            adapt_loss = self.compute_loss(batch)
            adapt(adapt_loss)
        
        #此处为外循环即使用了支持集所有的批次去模拟训练过程以此得到该任务的最终模型
        #现在需要在eval_batch上计算外循环损失，由于是外循环，所以这个eval_batch的提取方式是next(iter(eval_batch))，一次仅取出一个batch用于计算损失，而不是像adapt_batch的构造方式一样一次性取出所有的该任务的支持集batch
        if training: # compute loss for training, eval_batch should be ranking dataset
            output = self.compute_loss(eval_batch)
        else: # make predictions for testing, eval_batch should be regular dataset
            with torch.no_grad():
                self.model.eval()
                output = self.predict(eval_batch)[0]
                self.model.train()
        
        if isinstance(self.model, PeftModel):
            replace_modules(self.model, self.adapter_name, self.adapter.module)
        else:
            self.model = backup
        return output
    
    def train_step(self, batch):
        loss = []
        self.optimizer.zero_grad()
        for adapt_batch, eval_batch in zip(batch['adapt_batches'], batch['eval_batches']):
            #fast_adapt执行内循环，通过compute_loss计算内循环的损失，使用adapt方法进行更新以得到新的模型
            #然后假如是traing，则计算外循环的损失
            eval_loss = self.fast_adapt(adapt_batch, eval_batch)
            if not eval_loss.isfinite():
                continue
            #此处计算的才是真正的更新元学习器的参数
            eval_loss.backward()
            loss.append(eval_loss.item())
        if len(loss) == 0:
            return 0.
        
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1. / len(loss))
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        #返回这个step的损失
        return sum(loss) / len(loss)
    
    #调用fast_adapt方法不启用training模式直接得到predict的结果
    def adapt_predict(self, batch):
        predicts = []
        for adapt_batch, eval_batch in zip(batch['adapt_batches'], batch['eval_batches']):
            eval_preds = self.fast_adapt(adapt_batch, eval_batch,  training=False)
            predicts.append(eval_preds)
        #长度为num_batch的列表，每个元素为batch_size,1的score
        return predicts
    
    def compute_metrics(self, predicts, targets, labels):
        logs = defaultdict(float)
        for batch_preds, batch_tgts, batch_labels in zip(predicts, targets, labels):
            log = super().compute_metrics(batch_preds, batch_tgts, batch_labels)
            for key, value in log.items():
                logs[key] += value
        
        for key in logs.keys():
            logs[key] /= len(predicts)
        return logs
    
    #元学习则重写evalute_epoch方法，其中eval_iter是来源于目标蛋白质的由超参数-ts所划分出来数据集，并在此之上以0.5或0.75的比例进一步分割为支持集和查询集
    def evaluate_epoch(self, eval_iter, is_active=False):
        self.model.train()
        predicts = []
        logs = defaultdict(float)
        pbar = tqdm(eval_iter, desc='Evaluating')
        #这里pbar的大小是由支持集大小决定的，等于支持集大小/-meb
        for batch in pbar:
            #得到一个列表，长度批量个数，元素为该batch下各record的score(batch_size,1)
            batch_preds = self.adapt_predict(batch)
            batch_preds = [preds.to('cpu') for preds in batch_preds]
            #长度为批量个数
            predicts.append(batch_preds)
            batch_tgts = [eval_batch['targets'].to('cpu') for eval_batch in batch['eval_batches']]
            batch_labels = [eval_batch['labels'].to('cpu') for eval_batch in batch['eval_batches']]
            
            log = self.compute_metrics(batch_preds, batch_tgts, batch_labels)
            pbar.set_postfix({self.eval_metric: log[self.eval_metric]})
            for key, value in log.items():
                logs[key] += value
        
        predicts = [torch.cat(preds) for preds in zip(*predicts)]
        for key in logs.keys():
            logs[key] /= len(eval_iter)
            print('{}: {:.3f}'.format(key, logs[key]))
        return predicts, logs
               
