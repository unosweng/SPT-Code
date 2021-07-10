from transformers import BartModel, BartForConditionalGeneration, BartConfig, Seq2SeqTrainingArguments, \
    IntervalStrategy, SchedulerType, BartForSequenceClassification

import logging
import os

import vars
from data.dataset import CodeDataset
from data.vocab import Vocab
from utils.general import count_params, human_format, layer_wise_parameters
from utils.trainer import CodeTrainer
from utils.callbacks import LogStateCallBack
from models.bart import BartForClassificationAndGeneration

logger = logging.getLogger(__name__)


def pre_train(args, tasks=None):

    if tasks is None:
        # tasks = [vars.TASK_CODE_AST_PREDICTION, vars.TASK_NEXT_CODE_PREDICTION, vars.TASK_METHOD_NAME_PREDICTION]
        tasks = [vars.TASK_NEXT_CODE_PREDICTION, vars.TASK_METHOD_NAME_PREDICTION]

    logger.info('*' * 100)
    logger.info('Initializing pre-training environments')

    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading and parsing datasets')
    dataset = CodeDataset(args, mode='pre_train')
    logger.info(f'The size of pre_training set: {len(dataset)}')
    logger.info('Datasets loaded and parsed successfully')

    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Building vocabularies')
    # code vocab
    code_vocab = Vocab(name=args.code_vocab_name, method=args.code_tokenize_method, vocab_size=args.code_vocab_size,
                       datasets=[dataset.codes], ignore_case=True, save_root=args.vocab_root)
    logger.info(f'The size of code vocabulary: {len(code_vocab)}')
    # ast vocab
    ast_vocab = Vocab(name=args.ast_vocab_name, method='word', datasets=[dataset.asts],
                      ignore_case=True, save_root=args.vocab_root)
    logger.info(f'The size of ast vocabulary: {len(ast_vocab)}')
    # nl vocab
    nl_vocab = Vocab(name=args.nl_vocab_name, method=args.nl_tokenize_method, vocab_size=args.nl_vocab_size,
                     datasets=[dataset.names],
                     ignore_case=True, save_root=args.vocab_root)
    logger.info(f'The size of nl vocabulary: {len(nl_vocab)}')

    logger.info('Vocabularies built successfully')

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Building model')
    config = BartConfig(vocab_size=len(code_vocab) + len(ast_vocab) + len(nl_vocab),
                        max_position_embeddings=1024,
                        encoder_layers=args.n_layer,
                        encoder_ffn_dim=args.d_ff,
                        encoder_attention_heads=args.n_head,
                        decoder_layers=args.n_layer,
                        decoder_ffn_dim=args.d_ff,
                        decoder_attention_heads=args.n_head,
                        activation_function='gelu',
                        d_model=args.d_model,
                        dropout=args.dropout,
                        use_cache=True,
                        pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
                        bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                        eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                        is_encoder_decoder=True,
                        decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                        forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                        max_length=100,
                        min_length=1,
                        num_beams=args.beam_width,
                        num_labels=2)
    model = BartForClassificationAndGeneration(config)
    # log model statistic
    logger.info('Model trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # pre-train
    # --------------------------------------------------
    for task in tasks:
        logger.info('-' * 100)
        logger.info(f'Pre-training task: {task.upper()}')

        dataset.set_task(task)

        if task == vars.TASK_CODE_AST_PREDICTION:
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, 'cap'),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=1,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=args.pre_train_n_epoch,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, 'cap'),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.tensor_board_logging_steps,
                                                     save_strategy=IntervalStrategy.EPOCH,
                                                     seed=args.random_seed,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     dataloader_pin_memory=True)
            trainer = CodeTrainer(main_args=args,
                                  code_vocab=code_vocab,
                                  ast_vocab=ast_vocab,
                                  nl_vocab=nl_vocab,
                                  task='cap',
                                  model=model,
                                  args=training_args,
                                  data_collator=None,
                                  train_dataset=dataset,
                                  tokenizer=nl_vocab,
                                  model_init=None,
                                  compute_metrics=None,
                                  callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Start pre-training task: CAP')
            # model device
            logger.info('Device: {}'.format(next(model.parameters()).device))
            cap_result = trainer.train()
            logger.info(f'Pre-training task CAP finished')
            trainer.save_model(os.path.join(args.model_root, 'cap'))

        elif task == vars.TASK_NEXT_CODE_PREDICTION:
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, 'ncp'),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=1,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=args.pre_train_n_epoch,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, 'ncp'),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.tensor_board_logging_steps,
                                                     save_strategy=IntervalStrategy.EPOCH,
                                                     seed=args.random_seed,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     dataloader_pin_memory=True)
            trainer = CodeTrainer(main_args=args,
                                  code_vocab=code_vocab,
                                  ast_vocab=ast_vocab,
                                  nl_vocab=nl_vocab,
                                  task='ncp',
                                  model=model,
                                  args=training_args,
                                  data_collator=None,
                                  train_dataset=dataset,
                                  tokenizer=nl_vocab,
                                  model_init=None,
                                  compute_metrics=None,
                                  callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Start pre-training task: NCP')
            # model device
            logger.info('Device: {}'.format(next(model.parameters()).device))
            ncp_result = trainer.train()
            logger.info('Pre-training task NCP finished')
            trainer.save_model(os.path.join(args.model_root, 'ncp'))

        elif task == vars.TASK_METHOD_NAME_PREDICTION:
            # --------------------------------------------------
            # trainer
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Initializing the running configurations')
            training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.pre_train_output_root, 'mnp'),
                                                     overwrite_output_dir=True,
                                                     do_train=True,
                                                     per_device_train_batch_size=args.batch_size,
                                                     gradient_accumulation_steps=1,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.lr_decay_rate,
                                                     max_grad_norm=args.grad_clipping_norm,
                                                     num_train_epochs=args.pre_train_n_epoch,
                                                     lr_scheduler_type=SchedulerType.LINEAR,
                                                     warmup_steps=args.warmup_steps,
                                                     logging_dir=os.path.join(args.tensor_board_root, 'mnp'),
                                                     logging_strategy=IntervalStrategy.STEPS,
                                                     logging_steps=args.tensor_board_logging_steps,
                                                     save_strategy=IntervalStrategy.EPOCH,
                                                     seed=args.random_seed,
                                                     dataloader_drop_last=False,
                                                     run_name=args.model_name,
                                                     load_best_model_at_end=True,
                                                     ignore_data_skip=False,
                                                     label_smoothing_factor=args.label_smoothing,
                                                     dataloader_pin_memory=True)
            trainer = CodeTrainer(main_args=args,
                                  code_vocab=code_vocab,
                                  ast_vocab=ast_vocab,
                                  nl_vocab=nl_vocab,
                                  task='mnp',
                                  model=model,
                                  args=training_args,
                                  data_collator=None,
                                  train_dataset=dataset,
                                  tokenizer=nl_vocab,
                                  model_init=None,
                                  compute_metrics=None,
                                  callbacks=[LogStateCallBack()])
            logger.info('Running configurations initialized successfully')

            # --------------------------------------------------
            # train
            # --------------------------------------------------
            logger.info('-' * 100)
            logger.info('Start pre-training task: MNP')
            # model device
            logger.info('Device: {}'.format(next(model.parameters()).device))
            mnp_result = trainer.train()
            logger.info('Pre-training task MNP finished')
            trainer.save_model(os.path.join(args.model_root, 'MNP'))

    logger.info('Pre-training finished')
    model.save()

    return model, (code_vocab, ast_vocab, nl_vocab)