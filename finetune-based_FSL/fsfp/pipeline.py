import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from math import ceil
from itertools import chain
from transformers import EsmTokenizer, EsmForMaskedLM
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from . import config
from .trainer import RankingTrainer, MetaRankingTrainer, ContrastiveTrainer, MetaContrastiveTrainer
from .dataset.base import MutantSequenceData, RankingSequenceData, MetaRankingSequenceData, MetaContrastiveSequenceData
from .utils.data import make_dir, split_data
from .utils.score import metrics, group_scores, summarize_scores
import sys, os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

def print_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f'Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)')
    print(f'All params: {all_param}')

class Pipeline():
    def __init__(self, parsed_args, data_constructor=MutantSequenceData,
                 lora_modules=config.lora_modules, score_fn=None):
        if parsed_args.n_sites == [0]:
            parsed_args.n_sites = None
        if not 0 < parsed_args.train_size < 1:
            parsed_args.train_size = int(parsed_args.train_size)
        self.args = parsed_args
        self.device = 'cpu' if parsed_args.force_cpu or not torch.cuda.is_available() else 'cuda'
        self.data_constructor = data_constructor
        self.lora_modules = lora_modules
        self.score_fn = score_fn
        self.get_cv_size = lambda train: 0.75 if len(train['df']) > 50 else 0.5
        set_seed(parsed_args.seed)
    
    def get_base_model(self, load_dir=None):
        args = self.args
        model_name = config.model_dir[args.model]
        if load_dir is None:
            model = EsmForMaskedLM.from_pretrained(model_name)
            for name, param in model.named_parameters():
                if 'contact_head.regression' in name:
                    param.requires_grad = False
        else:
            model = EsmForMaskedLM.from_pretrained(load_dir)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        return model, tokenizer
        
    def get_save_dir(self, prefix, protein_name, prediction=False):
        args = self.args
        save_dir = '{}/{}/{}/{}/r{}{}{}{}{}{}{}{}{}'.format(
            config.pred_dir if prediction else config.ckpt_dir,
            prefix,
            args.model,
            protein_name,
            args.lora_r,
            f'_ts{args.train_size}_cv{args.cross_validation}' if not (args.augment and prefix == 'finetune') else '',
            f'_{args.retr_metric}_mt{args.meta_tasks}' if 'meta' in prefix else '',
            '_' + '-'.join(args.augment) if args.augment else '',
            '_regr' if args.list_size == 1 and (prefix == 'meta-transfer' or prefix == 'finetune') else '',
            '_ms' if args.n_sites != [1] else '',
            args.save_postfix,
            '_contrast' if args.use_contrast else '',
            '_msa' if args.use_msa and prediction else '')
        return save_dir

    def finetune_single(self, train, valid, save_dir=None, use_contrast=False, wt_logit_reg=None):
        args = self.args
        #如果含transfer表明进行迁移学习，加载已经保存的日志，并加载与训练好的模型
        if 'transfer' in args.mode:
            load_dir = self.get_save_dir('finetune' if args.mode == 'transfer' else 'meta', train['name'])
            logs = torch.load(load_dir + '/logs.pkl')
            if args.lora_r == 0:
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
            print(f'----------------------Continue training from epoch {logs["best_epoch"]}----------------------')
        #否则是finetune，初始化一个peft_model进行训练
        else:
            model, tokenizer = self.get_base_model()
            if args.lora_r > 0:
                lora_config = LoraConfig(r=args.lora_r,
                                    lora_alpha=args.lora_r,
                                    target_modules=self.lora_modules,
                                    lora_dropout=0.1,
                                    bias='none')
                model = get_peft_model(model, lora_config)
                print_trainable_params(model)
        
        #if use_contrast:
            #model_reg, _ = self.get_base_model()
            #for pm in model_reg.parameters():
                #pm.requires_grad = False
            #model_reg.eval()    #regularization model
        train_data = RankingSequenceData(train, tokenizer,
                                            mask=args.mask in {'train', 'all'},
                                            list_size=args.list_size,
                                            max_size=args.max_iter * args.train_batch,
                                            constructor=self.data_constructor,
                                            device=self.device)
        '''
        train_data = MutantSequenceData(train, tokenizer,
                                        mask=args.mask in {'train', 'all'},
                                        device=self.device)
        '''
        train_iter = DataLoader(train_data,
                                batch_size=args.train_batch,
                                shuffle=True,
                                collate_fn=train_data.collate)
        if not use_contrast:
            trainer = RankingTrainer(model.to(self.device),
                                    optimizer=args.optimizer,
                                    lr=args.learning_rate,
                                    epochs=args.epochs,
                                    max_grad_norm=args.max_grad_norm,
                                    score_fn=self.score_fn,
                                    eval_metric=args.eval_metric,
                                    log_metrics=metrics,
                                    save_dir=save_dir,
                                    patience=args.patience)
        else:
            trainer = ContrastiveTrainer(model=model.to(self.device),
                                    #model_reg=model_reg.to(self.device),
                                    optimizer=args.optimizer,
                                    lr=args.learning_rate,
                                    epochs=args.epochs,
                                    max_grad_norm=args.max_grad_norm,
                                    score_fn=self.score_fn,
                                    eval_metric=args.eval_metric,
                                    log_metrics=metrics,
                                    save_dir=save_dir,
                                    patience=args.patience)
        
        report = {}
        if valid is not None and args.cross_validation > 0:
            eval_data = self.data_constructor(valid, tokenizer,
                                              mask=args.mask in {'eval', 'all'},
                                              device=self.device)
            eval_iter = DataLoader(eval_data,
                                   batch_size=args.eval_batch,
                                   collate_fn=eval_data.collate)
            print('Computing zero-shot scores...')
            #预先计算没有微调的版本的统计指标作为baseline
            _, report['baseline'] = trainer.evaluate_epoch(eval_iter)
        else:
            eval_iter = None
        logs = trainer(train_iter, eval_iter)
        report.update(logs)
        report['best_epoch'] = trainer.best_epoch
        #返回一个字典，键为baseline（未微调模型的三个统计指标），lr、train_loss、三个统计指标（均为长度是训练周期数的列表），best_epoch
        return report
    
    def finetune_single_cv(self, train, test=None, use_contrast=False):
        args = self.args
        save_dir = self.get_save_dir(args.mode, train['name'])
        if os.path.exists(save_dir + '/logs.pkl'):
            print(f'{save_dir}/logs.pkl already exist, skipping!')
            report = torch.load(save_dir + '/logs.pkl')
            return report
        if args.cross_validation <= 1:
            report = self.finetune_single(train, test, save_dir, use_contrast)
            torch.save(report, save_dir + '/logs.pkl')
            return report
        
        cv_size = self.get_cv_size(train)
        splits = [split_data(train, cv_size, True) for _ in range(args.cross_validation)]
        epochs = args.epochs
        for i, (cv_train, cv_valid) in enumerate(splits):
            print(f'======================Cross validation: Split {i + 1}======================')
            #取出其中的一折进行微调训练，得到report
            #含有lr，train_loss，评估指标如spearman的字典，值为每个epoch下对应键的数值以及最佳周期
            cv_report = self.finetune_single(cv_train, cv_valid, use_contrast=use_contrast)
            #使用len(cv_report[args.eval_metric]获取实际的epoch运行数量,并将epoch固定在最少的伦次
            args.epochs = min(args.epochs, len(cv_report[args.eval_metric]))
            if i == 0:
                #第1折直接复制给report
                report = cv_report
                continue
            #将当前折baseline存的三个指标加和到baseline指定的指标上
            for key, value in cv_report['baseline'].items():
                report['baseline'][key] += value
            #将当前折记录的长为epoch的三项指标的列表存储的数值按位加到大的report上
            for key in metrics:
                for i in range(args.epochs):
                    report[key][i] += cv_report[key][i]
        #baseline取平均
        for key in report['baseline'].keys():
            report['baseline'][key] /= len(splits)
        #截止到最新的epoch取平均
        for key in metrics:
            report[key] = [value / len(splits) for value in report[key][:args.epochs]]
            #依据关注的指标找到在k折上表现最好的周期以及该分数
            if key == args.eval_metric: # find best epoch based on cv scores
                best_epoch, best_score = max(enumerate(report[key]), key=lambda x: x[1])
                best_epoch += 1
        print(f'CV-estimated best validating {args.eval_metric} reached at epoch {best_epoch}: {best_score:.3f}')
        print(f'----------------------Training on full data for {best_epoch} epochs----------------------')
        #新增一个best_epoch用于存储最佳的训练周期数，并更改args.epochs使得 self.finetune_single只训练指定周期
        report['best_epoch'] = args.epochs = best_epoch
        #在全体数据train（此处没用通过cv分割成k折）进行训练
        #由于无valid，所以返回的logs里面没有bseline，只含lr，train_loss的列表，长度为args.epochs，还有best_epoch
        logs = self.finetune_single(train, None, save_dir, use_contrast)
        #将train_loss更新为实际上在全数据上训练的loss
        report['train_loss'] = logs['train_loss']
        torch.save(report, save_dir + '/logs.pkl')
        #复原超参
        args.epochs = epochs
        return report
    
    def meta_single(self, train, eval_train, eval_test=None, use_contrast=False):
        args = self.args
        model, tokenizer = self.get_base_model()
        if args.lora_r > 0:
            lora_config = LoraConfig(r=args.lora_r,
                                lora_alpha=args.lora_r,
                                target_modules=self.lora_modules,
                                lora_dropout=0.1,
                                bias='none')
            model = get_peft_model(model, lora_config)
            print_trainable_params(model)
        
        #if use_contrast:
            #model_reg, _ = self.get_base_model()
        
        #将元任务训练集(选出的topk个)对半分，为一个列表含topk个元组，每个元组为一个database被对半分后的两个子集（字典）
        train_splits = [split_data(protein, 0.5, True) for protein in train]

        #返回一个字典形式的结果，键为adapt_batches和eval_batches，这两个均为列表，长度为支持集和查询集的大小
        train_data = MetaRankingSequenceData(train_splits, tokenizer,
                                                adapt_batch_size=args.meta_train_batch,
                                                eval_batch_size=args.meta_eval_batch,
                                                adapt_steps=args.adapt_steps,
                                                mask=args.mask,
                                                list_size=args.list_size,
                                                training=True,
                                                constructor=self.data_constructor,
                                                device=self.device)
        '''
        train_data = MetaContrastiveSequenceData(train_splits, tokenizer,
                                                adapt_steps=args.adapt_steps,
                                                mask=args.mask,
                                                training=True,
                                                constructor=self.data_constructor,
                                                device=self.device)
        '''
        #得到指定批量大小的支持集和查询集
        #假如是在元学习过程由于-tb设置为1，故一次取出一个元任务进行学习
        train_iter = DataLoader(train_data,
                                batch_size=args.train_batch,
                                shuffle=True,
                                collate_fn=train_data.collate)
        
        save_dir = self.get_save_dir(args.mode, eval_train['name'])
        if os.path.exists(save_dir + '/logs.pkl'):
            print(f'{save_dir}/logs.pkl already exist, skipping!')
            report = torch.load(save_dir + '/logs.pkl')
            return report
        if not use_contrast:
            trainer = MetaRankingTrainer(model.to(self.device),
                                        optimizer=args.optimizer,
                                        lr=args.learning_rate,
                                        epochs=args.epochs,
                                        max_grad_norm=args.max_grad_norm,
                                        score_fn=self.score_fn,
                                        adapt_lr=args.adapt_lr,
                                        eval_metric=args.eval_metric,
                                        log_metrics=metrics,
                                        save_dir=save_dir,
                                        patience=args.patience)
        else:
            trainer = MetaContrastiveTrainer(model.to(self.device),
                                        #model_reg=model_reg.to(self.device),
                                        optimizer=args.optimizer,
                                        lr=args.learning_rate,
                                        epochs=args.epochs,
                                        max_grad_norm=args.max_grad_norm,
                                        score_fn=self.score_fn,
                                        adapt_lr=args.adapt_lr,
                                        eval_metric=args.eval_metric,
                                        log_metrics=metrics,
                                        save_dir=save_dir,
                                        patience=args.patience)
        
        report = {}
        if args.cross_validation > 0:
            if args.cross_validation == 1:
                eval_splits = [(eval_train, eval_test)]
            else:
                #根据eval_train数据集的大小确定cv的比例是0.5还是0.75
                #假如可用的数据特别少，那么选取0.5作为训练集，否则选取0.75作为训练集
                #这个可用的数据是由-ts超参数决定的，因为eval_train和eval_test就是由这个参数划分的
                cv_size = self.get_cv_size(eval_train)
                #eval_train 和 eval_test是目标蛋白已经通过split_data划分出来的数据集
                #此处获得cv个data用于交叉验证
                eval_splits = [split_data(eval_train, cv_size, True) for _ in range(args.cross_validation)]
            eval_data = MetaRankingSequenceData(eval_splits, tokenizer,
                                                    adapt_batch_size=args.meta_train_batch,
                                                    eval_batch_size=args.eval_batch,
                                                    adapt_steps=args.adapt_steps,
                                                    mask=args.mask,
                                                    list_size=args.list_size,
                                                    training=False,
                                                    constructor=self.data_constructor,
                                                    device=self.device)
            '''
            eval_data = MetaContrastiveSequenceData(eval_splits, tokenizer,
                                                    adapt_steps=args.adapt_steps,
                                                    mask=args.mask,
                                                    training=False,
                                                    constructor=self.data_constructor,
                                                    device=self.device)
            '''
            #由于eval_iter是一个batch一个batch的，所以batch_size=1
            #cv=5,batch_size=1,所以evaluating显示的batch数为5
            eval_iter = DataLoader(eval_data,
                                   batch_size=1,
                                   collate_fn=eval_data.collate)
            #调用evaluate_epoch方法返回查询集的预测结果
            #此处就是无元迁移学习，仅使用小样本微调的结果
            #一个train_epoch或evaluate_epoch本质就是在完备的支持集上进行微调之后使用全体的的查询集进行外循环的损失统计更新元学习器
            _, report['baseline'] = trainer.evaluate_epoch(eval_iter)
        else:
            eval_iter = None
        #此处启用了元学习的策略：先使用train_iter进行元学习优化元学习器
        #然后在eval_iter进行目标任务的微调，使用目标蛋白的train和eval进行微调
        #含有lr，train_loss，评估指标如spearman的字典，值为每个epoch下对应键的数值
        #由于辅助任务数小于4，三个任务x2得到六个所以在训练期间会有6个batch
        logs = trainer(train_iter, eval_iter)
        report.update(logs)
        #记录最佳表现出现的周期
        report['best_epoch'] = trainer.best_epoch
        torch.save(report, save_dir + '/logs.pkl')
        return report
    
    def test_single(self, train, test, use_contrast, use_msa=False, seq_aln_file=None):
        args = self.args
        if args.epochs > 0:
            load_dir = self.get_save_dir(args.mode, test['name'])
            if args.lora_r == 0:
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                #包装pLM至PEFT微调后的版本
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
        else:
            model, tokenizer = self.get_base_model()
        #构造一个regular dataset，mask为eval或者all时则生成num_record个序列，各自在突变位置进行掩码
        test_data = self.data_constructor(test, tokenizer,
                                          mask=args.mask in {'eval', 'all'},
                                          device=self.device)
        test_iter = DataLoader(test_data,
                               batch_size=args.eval_batch,
                               collate_fn=test_data.collate)
        if not use_contrast:
        #启用RankingTrainer
            trainer = RankingTrainer(model.to(self.device), log_metrics=[], score_fn=self.score_fn)
        else:
            
            trainer = ContrastiveTrainer(model=model.to(self.device), tokenizer=tokenizer, 
                                         log_metrics=[], score_fn=self.score_fn, use_msa=use_msa, 
                                         seq_aln_file=seq_aln_file, alpha=args.alpha)
        #调用TrainerBase的evaluate_epoch使用加载的模型去直接预测
        predicts, _ = trainer.evaluate_epoch(test_iter)
        #获得num_record，1的fitness score
        predicts = predicts.tolist()
        
        predicts = pd.Series(predicts, index=test['df'].index, name='prediction')
        #得到dataframe形式的report，存储着包括['single_local', 'single_cross', 'single_rest','multi_combined', 'multi_cross', 'multi_rest', 'all_rest']这些类别的统计数据
        #统计数据包括spearman，ngdc，topk_pr
        #行为不同组别，列为三个指标的df
        report, _ = group_scores(train['df'], predicts, test['df'])
        print('======================Breakdown results======================')
        print(report)
        
        print('Saving model predictions...')
        save_path = self.get_save_dir(args.mode, test['name'], prediction=True)
        save_path += '_base.csv' if args.epochs == 0 else '.csv'
        make_dir(save_path)
        predicts.to_csv(save_path)
        return report
    
    def select_datasets(self, all_proteins):
        args = self.args
        if args.protein in all_proteins.keys():
            return all_proteins[args.protein]
        
        proteins = chain(*all_proteins.values())
        if args.train_size >= 1:
            proteins = filter(lambda x: len(x['df']) > args.train_size, proteins)
        
        if args.protein == 'all':
            return list(proteins)
        if args.protein == 'single-site':
            return list(filter(lambda x: x['n_sites'][-1] == 1, proteins))
        if args.protein == 'multi-site':
            return list(filter(lambda x: x['n_sites'][-1] > 1, proteins))
        if len(args.protein) == 2:
            proteins = list(proteins)
            N, i = int(args.protein[0]), int(args.protein[1])
            n = ceil(len(proteins) / N)
            j = (i - 1) * n
            return proteins[j:j + n]
    
    def get_meta_database(self, all_proteins):
        args = self.args
        database = {name: max(datasets, key=lambda x: len(x['df'])) \
                        for name, datasets in all_proteins.items()}
        topk = torch.load(f'{config.retr_dir}/topk_{args.model}_{args.retr_metric}.pkl')
        return database, topk
    
    def augment_data(self, protein):
        args = self.args
        #如果为adaptive，则加载自己提供的私人数据集（形式同proteingym），提取目标蛋白对应的
        if args.augment == ['adaptive']:
            aug_models = pd.read_csv(f'{config.retr_dir}/aug_models{args.save_postfix}.csv', index_col=0)
            #
            aug_models = [aug_models.loc[protein['name'], args.train_size]]
        else:
            aug_models = args.augment
        #aug_models应当是一个列表，列表里存了要启用的模型的名字，通过超参数指定
        #从跟proteingym蛋白一样的文件夹中的指定蛋白质文件中取出该蛋白质的该模型fitness预测结果
        raw_data = pd.read_csv(f'{config.raw_data_dir}/{protein["name"]}.csv', index_col='mutant',
                               usecols=aug_models + ['mutant'])
        aug_data = []
        for model_name in aug_models:
            new = deepcopy(protein)
            #将DMS_score替换为指定model的fitness预测结果
            new['df']['DMS_score'] = raw_data[model_name]
            if new['n_sites'][-1] > 2:
                new, _ = split_data(new, len(new['df']), n_sites=[1, 2])
                new["df"] = new["df"].dropna(subset=["DMS_score"])
            aug_data.append(new)
        return aug_data
    
    def __call__(self, all_proteins):
        args = self.args
        #从merged.pkl文件加载指定蛋白质的数据：双层字典，外层键为蛋白名字值为内层字典，内层字典键为wt_seq，df，offset, n_sites, name
        proteins = self.select_datasets(all_proteins)
        
        #如果是meta learning则加载已经先前计算的余弦相似度最近的topk数据集（topk），以及全部database
        if args.mode == 'meta':
            database, topk = self.get_meta_database(all_proteins)
            
        reports = {}
        for protein in proteins:
            print(f'**********************Current dataset: {protein["name"]}**********************')
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                eval_metric = args.eval_metric
                args.eval_metric = 'ndcg' # in case of nan spearmanr
            
            #依据train_size(0-1的一个小数或指定few shot的个数)划分数据集
            #设置了ts为40，在字典中的df层面选出了40条record作为train其余的为test
            #此处的train和test为当前任务的训练集和测试集
            train, test = split_data(protein, args.train_size, n_sites=args.n_sites,
                                     neg_train=args.negative_train, scale=args.list_size == 1)
            if args.test:
                #得到dataframe形式的report，存储着包括['single_local', 'single_cross', 'single_rest','multi_combined', 'multi_cross', 'multi_rest', 'all_rest']这些类别的统计数据
                #统计数据包括spearman，ngdc，topk_pr
                seq_aln_file = f"{args.seq_aln_dir}/{protein['name']}.a2m"
                report = self.test_single(train=train, test=test, use_contrast=args.use_contrast, 
                                          use_msa=args.use_msa, seq_aln_file=seq_aln_file)
                    
            elif args.mode != 'meta':
                if args.mode == 'finetune' and args.augment:
                    #只返回第一个启用模型的数据
                    protein = self.augment_data(protein)[0]
                #使用self.finetune_single_cv函数，在cv上先找到最佳微调伦次，应用该轮次在train上训练
                #此处返回的report没有任何用，只是这个函数的输出而已，因为目的是训练只要保存了report即可
                report = self.finetune_single_cv(train, test, use_contrast=args.use_contrast)
            else:
                src_name = '_'.join(protein['name'].split('_')[:2])
                #根据query：src_name取出topk个proteingym数据集
                tgt_names = topk[src_name]['tgt_names'][:args.meta_tasks]
                #根据tgt_names从database中取出元任务训练集
                meta_train = [database[name] for name in tgt_names]

                if args.augment:
                    #将增强数据替换掉proteingym数据
                    meta_train[-len(args.augment):] = self.augment_data(protein)
                if args.meta_tasks < 4:
                    meta_train *= 2
                #meta_train为一个列表列表里面每个元素为一个字典即被选出的topk个数据集
                #train和test则是依据split_data在这个目标蛋白数据集上划分出来的数据集
                #返回一个字典，键为lr，train_loss，评估指标，最佳表现的周期,baseline
                report = self.meta_single(meta_train, train, test, args.use_contrast)
            
            reports[protein['name']] = report
            torch.cuda.empty_cache()
            
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                args.eval_metric = eval_metric
        
        if args.test and args.protein in {'single-site', 'multi-site', 'all'}:
            save_path = self.get_save_dir(args.mode, args.protein, prediction=True)
            save_path += '_base.pkl' if args.epochs == 0 else '.pkl'
            make_dir(save_path)
            reports = summarize_scores(reports, save_path)
            print('**********************Score summary**********************')
            #print(reports[args.eval_metric])
            print(reports['spearmanr'])
            print('----------------------Separator----------------------')
            print(reports['ndcg'])
            print('----------------------Separator----------------------')
            print(reports['topk_pr'])
