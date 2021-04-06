import sys
sys.path.insert(1, 'pair_extraction')
from trainer import *
import time

def predict_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance]):
    ## evaluation
    tp, fp, tn, fn = 0, 0, 0, 0
    # metrics, metrics_e2e = np.asarray([0, 0, 0], dtype=int), np.asarray([0, 0, 0], dtype=int)
    metrics, metrics_e2e = np.asarray([0, 0, 0], dtype=int), np.zeros((1, 3), dtype=int)
    pair_metrics = np.asarray([0, 0, 0], dtype=int)
    batch_idx = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        processed_batched_data = simple_batching(config, batch)
        batch_max_scores, batch_max_ids, pair_ids = model.decode(processed_batched_data)

        metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, processed_batched_data[-6], processed_batched_data[2], config.idx2labels)
        metrics_e2e += evaluate_batch_insts_e2e(one_batch_insts, batch_max_ids, processed_batched_data[-6], processed_batched_data[2], config.idx2labels, processed_batched_data[-8], pair_ids, processed_batched_data[-1])
        #word_seq_lens = processed_batched_data[2].tolist()
        """for batch_id in range(batch_max_ids.size()[0]):
            length = word_seq_lens[batch_id]

            gold = processed_batched_data[-6][batch_id][:length]

            s_id = (gold == 2).nonzero()
            b_id = (gold == 3).nonzero()
            e_id = (gold == 4).nonzero()
            i_id = (gold == 5).nonzero()
            gold_id = torch.cat([s_id, b_id, e_id, i_id]).squeeze(1)
            gold_id, _ = gold_id.sort(0, descending=False)
            gold_id = gold_id[gold_id < processed_batched_data[-1][batch_id]]

            # argu_id = torch.LongTensor(list(set(gold_id.tolist()).intersection(set(pred_id.tolist()))))
            argu_id = torch.LongTensor(list(set(gold_id.tolist())))
            # print('gold_id', gold_id, 'pred_id', pred_id, 'argu_id', argu_id)

            # print(pair_ids[batch_id].size(), batch[-3][batch_id].size())
            one_batch_insts[batch_id].gold2 = processed_batched_data[-3][batch_id].tolist()
            one_batch_insts[batch_id].pred2 = pair_ids[batch_id].squeeze(2).tolist()


            # print(one_batch_insts[batch_id].gold2)
            # print(torch.sum(one_batch_insts[batch_id].pred2, dim=1))

            # pred2 = one_batch_insts[batch_id].pred2[argu_id]
            pred2 = pair_ids[batch_id].squeeze(2)
            # gold2 = one_batch_insts[batch_id].gold2[argu_id]
            gold2 = processed_batched_data[-3][batch_id]


            # print('argu_id:  ',argu_id.size(),argu_id)
            # print('one_batch_insts[batch_id].pred2:  ',one_batch_insts[batch_id].pred2.size(),one_batch_insts[batch_id].pred2)

            gold_pairs = gold2.flatten()
            pred_pairs = pred2.flatten()

            # print(gold_pairs,pred_pairs)
            sum_table = gold_pairs + pred_pairs
            # print(sum_table.size(),sum_table[:100])
            sum_table_sliced = sum_table[sum_table >= 0]
            # print(sum_table_sliced.size(),sum_table_sliced)
            tp_tmp = len(sum_table_sliced[sum_table_sliced == 2])
            tn_tmp = len(sum_table_sliced[sum_table_sliced == 0])
            tp += tp_tmp
            tn += tn_tmp
            ones = len(gold_pairs[gold_pairs == 1])
            zeros = len(gold_pairs[gold_pairs == 0])
            fp += (zeros - tn_tmp)
            fn += (ones - tp_tmp)
            # print(tp,tp_tmp,tn,tn_tmp,ones,zeros,fp,fn)"""


        batch_idx += 1
    """print('tp, fp, fn, tn: ', tp, fp, fn, tn)
    precision_2 = 1.0 * tp / (tp + fp) * 100 if tp + fp != 0 else 0
    recall_2 = 1.0 * tp / (tp + fn) * 100 if tp + fn != 0 else 0
    f1_2 = 2.0 * precision_2 * recall_2 / (precision_2 + recall_2) if precision_2 + recall_2 != 0 else 0
    acc = 1.0 *(tp+tn)/(fp+fn+tp+tn) * 100 if fp+fn+tp+tn!=0 else 0
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    p_e2e, total_predict_e2e, total_entity_e2e = metrics_e2e[:, 0], metrics_e2e[:, 1], metrics_e2e[:, 2]
    # precision_e2e = p_e2e * 1.0 / total_predict_e2e * 100 if total_predict_e2e != 0 else 0
    # recall_e2e = p_e2e * 1.0 / total_entity_e2e * 100 if total_entity_e2e != 0 else 0
    # fscore_e2e = 2.0 * precision_e2e * recall_e2e / (precision_e2e + recall_e2e) if precision_e2e != 0 or recall_e2e != 0 else 0
    total_predict_e2e[total_predict_e2e == 0] = sys.maxsize
    total_entity_e2e[total_entity_e2e == 0] = sys.maxsize

    precision_e2e = p_e2e * 1.0 / total_predict_e2e * 100
    recall_e2e = p_e2e * 1.0 / total_entity_e2e * 100

    sum_e2e = precision_e2e + recall_e2e
    sum_e2e[sum_e2e == 0] = sys.maxsize
    fscore_e2e = 2.0 * precision_e2e * recall_e2e / sum_e2e

    print("Task1: ", p, total_predict, total_entity)
    # print("Overall: ", p_e2e, total_predict_e2e, total_entity_e2e)

    print("Task1: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
    print("Task2: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f, acc: %.2f" % (name, precision_2, recall_2, f1_2, acc), flush=True)
    percs=[0.9]
    #percs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    for i in range(len(percs)):
        print("Overall ", percs[i], ": [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision_e2e[i], recall_e2e[i], fscore_e2e[i]), flush=True)"""
    return [precision, recall, fscore, precision_2, recall_2, f1_2, acc, precision_e2e, recall_e2e, fscore_e2e]

def run_model(config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]):
    start_time = time.time()
    model = NNCRF(config)
    num_param=0
    for idx in list(model.parameters()):
        try:
            num_param+=idx.size()[0]*idx.size()[1]
        except:
            num_param+=idx.size()[0]
    print(num_param)
    optimizer = get_optimizer(config, model)


    #batched_data = batching_list_instances(config, train_insts)
    #dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    model_folder = config.model_folder
    res_folder = "pair_extraction/results"
    model_path = f"pair_extraction/model_files/{model_folder}/lstm_crf.m"
    config_path = f"pair_extraction/model_files/{model_folder}/config.conf"
    res_path = f"{res_folder}/{model_folder}.predict.results"
    no_incre_dev = 0

    print("Evaluating...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
    test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
    #test_metrics = predict_model(config, model, test_batches, "test", test_insts)
    end_time = time.time()
    print("Time is %.2fs" % (end_time - start_time), flush=True)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_path, test_insts)
    print("Finished")

def pair_inference(pred_file):
    """parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    config_path = f"model_files/{opt.model_folder}/config.conf"
    model_folder = opt.model_folder"""
    config_path = f"pair_extraction/model_files/english_model_glove/config.conf"
    model_folder = "english_model_glove"
    with open(config_path, 'rb') as f:
        conf = pickle.load(f)
    conf.model_folder = model_folder
    conf.test_file = pred_file

    reader = Reader(conf.digit2zero)
    #set_seed(opt, conf.seed)

    #devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.context_emb != ContextEmb.none:
        print('Loading the ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.context_emb.name + ".vec", trains)
        #load_elmo_vec(conf.dev_file + "." + conf.context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.context_emb.name + ".vec", tests)

    conf.use_iobes(tests)
    conf.map_insts_ids(tests)

    run_model(conf, conf.num_epochs, [], [], tests)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    config_path = f"model_files/{opt.model_folder}/config.conf"
    model_folder = opt.model_folder
    with open(config_path, 'rb') as f:
        conf = pickle.load(f)
    conf.model_folder = model_folder

    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    #trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.context_emb != ContextEmb.none:
        print('Loading the ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.context_emb.name + ".vec", tests)

    #conf.use_iobes(trains)
    conf.use_iobes(devs)
    conf.use_iobes(tests)
    #conf.build_label_idx(devs + tests)

    #conf.build_word_idx([], devs, tests)
    #conf.build_emb_table()

    #conf.map_insts_ids(trains)
    conf.map_insts_ids(devs)
    conf.map_insts_ids(tests)

    #print("num chars: " + str(conf.num_char))
    # print(str(config.char2idx))

    #print("num words: " + str(len(conf.word2idx)))
    # print(config.word2idx)
    run_model(conf, conf.num_epochs, [], devs, tests)
