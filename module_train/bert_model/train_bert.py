import torch
from module_dataset.preprocess_dataset.dataloader import load_squad_to_torch_dataset
from transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm_notebook as tqdm
import random
import numpy as np
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import f1_score, accuracy_score


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# modify code train task squad to zalo task
def train_squad(args, tokenizer, model):
    # open to new log file (need modify with logging but later)
    w_log_file = open(args.path_log_file, "a")

    if not args.load_data_from_pt:
        train_dataset, train_dataloader = load_squad_to_torch_dataset(args.path_input_train_data,
                                                                      tokenizer,
                                                                      args.max_seq_length,
                                                                      args.max_query_length,
                                                                      args.batch_size,
                                                                      is_training=True)
        torch.save(train_dataset, args.path_pt_train_dataset)

    else:
        train_dataset = torch.load(args.path_pt_train_dataset)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    w_log_file.write("Number train sample: {}\n".format(len(train_dataset)))
    w_log_file.write("Load dataset done !!!\n")
    print("Load dataset done !!!")

    if not args.no_cuda:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    args.device = device

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    w_log_file.write("***** Running training *****\n")
    w_log_file.write("  Num examples = {}".format(len(train_dataset)))
    w_log_file.write("  Num Epochs = {}".format(args.num_train_epochs))
    w_log_file.write("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    w_log_file.write("  Total optimization steps = {}".format(t_total))

    n_epoch = 0
    global_step = 0

    model.zero_grad()
    set_seed(args)

    for _ in range(args.num_train_epochs):
        l_full_target = []
        l_full_predict = []
        tr_loss, logging_loss = 0.0, 0.0

        epoch_iterator = tqdm(train_dataloader, desc="training ...", leave=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'label': batch[3]
                      }

            loss, l_predict, l_target = model.loss(inputs['input_ids'],
                                                   inputs['attention_mask'],
                                                   inputs['token_type_ids'],
                                                   inputs['label'])
            l_full_target.extend(l_target)
            l_full_predict.extend(l_predict)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step == len(train_dataloader) - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.save_steps > 0 and global_step % args.save_steps == 0 or \
                        step >= int(len(train_dataloader) - 1)) and \
                        global_step > (2/3 * (len(train_dataloader) / (args.batch_size * args.gradient_accumulation_steps))):
                    line_start_logging = "Log write at epoch: {}, step: {} and lr: {}\n".format(n_epoch,
                                                                                                global_step,
                                                                                                round(scheduler.get_lr()[0], 6))
                    print(line_start_logging)
                    w_log_file.write(line_start_logging)

                    f1_score_micro = f1_score(l_full_target, l_full_predict)
                    accuracy = accuracy_score(l_full_target, l_full_predict)

                    output_train = {
                        "loss": round(tr_loss / len(train_dataset), 3),
                        "accuracy": round(accuracy, 3),
                        "f1": round(f1_score_micro, 3)
                    }
                    line_log_train = "train result - loss: {}, acc: {}, f1: {}\n".format(output_train['loss'],
                                                                                         output_train['accuracy'],
                                                                                         output_train['f1'])
                    print(line_log_train)
                    w_log_file.write(line_log_train)

                    if args.path_input_test_data is not None:
                        w_log_file.write("Start evaluating test data !!\n")
                        output_test = evaluate(args, model, tokenizer, is_test=True)
                        line_log_test = "test result - loss: {}, acc: {}, f1: {}\n".format(output_test['loss'],
                                                                                           output_test['accuracy'],
                                                                                           output_test['f1'])
                        print(line_log_test)
                        w_log_file.write(line_log_test)

                    if args.path_input_validation_data is not None:
                        w_log_file.write("Start evaluating validation data !!\n")
                        output_validation = evaluate(args, model, tokenizer, is_test=False)
                        line_log_val = "test result - loss: {}, acc: {}, f1: {}\n".format(output_validation['loss'],
                                                                                          output_validation['accuracy'],
                                                                                          output_validation['f1'])
                        print(line_log_val)
                        w_log_file.write(line_log_val)
                    line_end_logging = "end for logging current step {} !!!".format(global_step)
                    print(line_end_logging)

                    w_log_file.write(line_end_logging)

                    prefix_dir_save = "epoch{}_step{}".format(n_epoch, global_step)
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, prefix_dir_save)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    w_log_file.write("Saving model checkpoint to {}".format(output_dir))

        n_epoch += 1
    w_log_file.close()


def evaluate(args, model, tokenizer, is_test=True):
    if is_test:
        path_input_data = args.path_input_test_data
        path_input_data_pt = args.path_input_test_pt
    else:
        path_input_data = args.path_input_validation_data
        path_input_data_pt = args.path_input_validation_pt

    if not args.load_data_from_pt:
        eval_dataset, eval_dataloader = load_squad_to_torch_dataset(path_input_data,
                                                                    tokenizer,
                                                                    args.max_seq_length,
                                                                    args.max_query_length,
                                                                    args.batch_size,
                                                                    is_training=True)
        torch.save(eval_dataset, path_input_data_pt)
    else:
        eval_dataset = torch.load(path_input_data_pt)
        test_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # Eval!
    print("***** Running evaluation")
    print("  Num examples = %d", len(eval_dataset))

    total_loss = 0.0
    l_full_predict = []
    l_full_target = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'label': batch[3]
                      }

            loss, l_predict, l_target = model.loss(inputs['input_ids'],
                                    inputs['attention_mask'],
                                    inputs['token_type_ids'],
                                    inputs['label'])
            total_loss += loss.item()
            l_full_predict.extend(l_predict)
            l_full_target.extend(l_target)

    f1_score_micro = f1_score(l_full_target, l_full_predict)
    accuracy = accuracy_score(l_full_target, l_full_predict)

    output_validation = {
        "loss": round(total_loss / len(eval_dataset), 3),
        "accuracy": round(accuracy, 3),
        "f1": round(f1_score_micro, 3)
    }

    return output_validation

