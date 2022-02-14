import argparse
import os, sys
import mxnet as mx
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import numpy as np
from mxboard import SummaryWriter

# network_version = "Original"
# network_name = "ResNet"
# name = ' '.join([network_version, network_name, str(number_of_layers)])
# print(name)


data_set = "cifar10" # "imagenet"
number_of_layers = 20 # 34, 50
# units = [7, 6, 6]
# stage = 3
# num_filters = [16, 32, 64]
# num_hidden = 10
# use_bottle_neck = False

# imagenet - 34, 50
# cifar10 - 20, 32, 56
cifar10_layer_list = [20, 32, 44, 56, 110, 1201]

# for log
global_steps = 0
global_steps_eval = 0
best_ac = -1 
best_ce = -1
best_epoch = -1
best_batch = -1

###### config ######
def parse_args():

    parser = argparse.ArgumentParser(description='Training Original Resnet Network')

    parser.add_argument('--number_of_layers', type=int, default=number_of_layers, help='number of layers')
    parser.add_argument('--data_set', default=data_set, help='dataset')
    parser.add_argument('--gpus', default=None, help='gpus')
    parser.add_argument('--model_load_epoch', default=None, help='load the model on an epoch')
    parser.add_argument('--retrain', default=False, help='if true, continue training. else false, training at epoch 0')
    parser.add_argument('--frequent', type=int, default=10, help='frequency of logging')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--num-examples', type=int, default=50000, help='the number of training examples')
    parser.add_argument('--num-examples-eval', type=int, default=10000, help='the number of validation examples')
    parser.add_argument('--lr', type=float, default=0.1, help='initialization learning reate')
    parser.add_argument('--data-dir', type=str, default='/home/aiteam/mjjeon/original_resnet/data/cifar10/', help='the input data directory')
    parser.add_argument('--model-root', type=str, default='/home/aiteam/mjjeon/original_resnet/models', help='the model directory path')
    parser.add_argument('--log-root', type=str, default='/home/aiteam/mjjeon/original_resnet/logs', help='the log directory path')
    # parser.add_argument('--train-log-path', type=str, default='/home/aiteam/mjjeon/original_resnet/train_logs', help='train log path')
    # parser.add_argument('--eval-log-path', type=str, default='/home/aiteam/mjjeon/original_resnet/eval_logs', help='eval log path')
    parser.add_argument('--num-epoch', type=int, default=300, help='number of epoch')
    parser.add_argument('--units', type=str, default="", help='unit')
    p_args = parser.parse_args()

    return p_args


def set_config(args):

    if args.data_set == "imagenet":
        args.stage = 4
        args.num_filters = [64, 128, 256, 512]
        args.num_hidden = 1000
        args.num_examples = 1281167
    elif args.data_set == "cifar10":
        args.aug_level = 1
        args.stage = 3
        args.num_filters = [16, 32, 64]
        args.num_hidden = 10
        args.num_examples = 50000
    else:
        args.stage = 4
        args.num_filters = [64, 128, 256, 512]
        args.num_hidden = 1000

    if args.number_of_layers == 34:
        args.units = [3,4,6,3]
    elif args.number_of_layers == 50:
        args.units = [3,4,6,3]
    elif args.number_of_layers in cifar10_layer_list:
        # total layer : 6n + 2
        # number of layer   : 2n + 1, 2n, 2n
        # filter            : 16,     32, 64
        n = int((args.number_of_layers - 2)/6)
        args.units = [2*n + 1, 2*n, 2*n] 
        # e.g. number_of_layers == 20, n = 3, units = [7, 6, 6]
    else:
        print('please check number of layers...')
        sys.exit()
        

    if args.number_of_layers in [50, 101, 152]:
        args.use_bottle_neck = True
    if args.data_set == "cifar10":
        args.use_bottle_neck = False

    args.momentum = 0.9
    # args.lr = 
    args.wd = 0.0001 # weight decay

    # shortcut
    # A: zero padding used for increasing dimensions, all shortcuts are parameter free
    # B: projection used for increasing dim, other shortcuts are identity
    # C: all shortcuts are projections
    args.shortcut_type = "B"

    args.gpus = "0,1,2,3"



###### network ######
def resnet():

    data = mx.symbol.Variable('data')
    
    # conv1
    if args.data_set == "cifar10":
        data = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=args.num_filters[0], stride=(1, 1), pad=(1, 1), name='conv1')
    else:
        data = mx.symbol.Convolution(data=data, kernel=(7, 7), num_filter=args.num_filters[0], stride=(2, 2), pad=(3, 3), name='conv1')
        data = mx.symbol.Pooling(data=data, kernel=(3, 3), global_pool=False, pool_type='max', stride=(2, 2), pad=(1, 1), name='pool1')
    
    # conv2 maxpool
    body = data

    for i_unit in range(len(args.units)): # [3, 4, 6, 3] -> 0, 1, 2, 3

        for j_unit in range(args.units[i_unit]): # [3, 4, 6, 3] -> 0,1,2 / 0,1,2,3 / 0,1,2,3,4,5 / 0,1,2

            data = body

            if args.use_bottle_neck:
                if j_unit == 0 and i_unit != 0:
                    body = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(2, 2), pad=(0, 0), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_1") # conv2_0
                else:
                    # body = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(1, 1), pad=(1, 1), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_1") # conv2_0
                    body = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(1, 1), pad=(0, 0), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_1") # conv2_0

                body = mx.symbol.BatchNorm(data=body, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_1")
                body = mx.symbol.Activation(data=body, act_type='relu', name="relu" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_1")
                
                body = mx.symbol.Convolution(data=body, kernel=(3,3), num_filter=args.num_filters[i_unit], stride=(1, 1), pad=(1, 1), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_2") # conv2_0
                body = mx.symbol.BatchNorm(data=body, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_2")
                body = mx.symbol.Activation(data=body, act_type='relu', name="relu" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_2")

                body = mx.symbol.Convolution(data=body, kernel=(1,1), num_filter=args.num_filters[i_unit]*4, stride=(1, 1), pad=(0, 0), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_3") # conv2_0
                body = mx.symbol.BatchNorm(data=body, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_3")
                
                # shortcut 
                # j_unit == 0 and i_unit != 0 인 경우는 차원이 바뀌었을 경우 downsampling 해주기 위하여.
                if args.shortcut_type == "A":
                    if j_unit == 0 and i_unit != 0:
                        pass
                        # sc = mx.symbol.pad(data=data, pad_width=(), constant_value=0) # zero padding pad_width..
                elif args.shortcut_type == "B":
                    #if j_unit == 0 and i_unit != 0:
                    if j_unit == 0 and i_unit != 0:
                        sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit]*4, stride=(2, 2), pad=(0, 0), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_4")
                        sc = mx.symbol.BatchNorm(data=sc, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_4")
                    elif j_unit == 0:
                        # sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(2, 2), pad=(0, 0), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_4")
                        sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit]*4, stride=(1, 1), pad=(0, 0), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_4")
                        sc = mx.symbol.BatchNorm(data=sc, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_4")
                    
                    else:
                        sc = data
                elif args.shortcut_type == "C":
                    if j_unit == 0 and i_unit != 0:
                        sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(2, 2), pad=(0, 0), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_4")
                    else:
                        sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(1, 1), pad=(0, 0), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_4")
                    sc = mx.symbol.BatchNorm(data=sc, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_4")

                body = body + sc
                 
                body = mx.symbol.Activation(data=body, act_type='relu', name="relu" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_3")
            else:
                # batch norm에서 fixgamma=False 해주는 이유는 파라미터 갯수 줄이기 위해라고 추정, 논문 보면 gamma, beta 값 때문에 독립적인 변수들이 생기게 된다고 함.
                if j_unit == 0 and i_unit != 0:
                    body = mx.symbol.Convolution(data=data, kernel=(3,3), num_filter=args.num_filters[i_unit], stride=(2, 2), pad=(1, 1), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_1") # conv2_0
                else:
                    body = mx.symbol.Convolution(data=data, kernel=(3,3), num_filter=args.num_filters[i_unit], stride=(1, 1), pad=(1, 1), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_1") # conv2_0

                body = mx.symbol.BatchNorm(data=body, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_1")
                
                body = mx.symbol.Activation(data=body, act_type='relu', name="relu" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_1")
                
                body = mx.symbol.Convolution(data=body, kernel=(3,3), num_filter=args.num_filters[i_unit], stride=(1, 1), pad=(1, 1), name="conv" + str(i_unit + 2) + "_" + str(j_unit + 1) + "_2") # conv2_0
                body = mx.symbol.BatchNorm(data=body, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_2")
                
                # shortcut 
                # j_unit == 0 and i_unit != 0 인 경우는 차원이 바뀌었을 경우 downsampling 해주기 위하여.
                if args.shortcut_type == "A":
                    if j_unit == 0 and i_unit != 0:
                        pass
                        # sc = mx.symbol.pad(data=data, pad_width=(), constant_value=0) # zero padding pad_width..
                elif args.shortcut_type == "B":
                    if j_unit == 0 and i_unit != 0:
                        # padding = (0, 0)
                        sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(2, 2), pad=(0, 0), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1))
                        sc = mx.symbol.BatchNorm(data=sc, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_3")
                    else:
                        sc = data
                elif args.shortcut_type == "C":
                    if j_unit == 0 and i_unit != 0:
                        sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(2, 2), pad=(1, 1), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1))
                    else:
                        sc = mx.symbol.Convolution(data=data, kernel=(1,1), num_filter=args.num_filters[i_unit], stride=(1, 1), pad=(1, 1), name="sc" + str(i_unit + 2) + "_" + str(j_unit + 1))
                    sc = mx.symbol.BatchNorm(data=sc, momentum=args.momentum, fix_gamma=False, name="bn" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_3")

                body = body + sc
                 
                body = mx.symbol.Activation(data=body, act_type='relu', name="relu" + str(i_unit + 1) + "_" + str(j_unit + 1) + "_2")


    data = mx.symbol.Pooling(data=body, global_pool=True, pool_type='avg', name='pool2')
    # data = mx.symbol.Flatten(data=data, name='flat1')
    data = mx.symbol.flatten(data=data, name='flat1')
    data = mx.symbol.FullyConnected(data=data, num_hidden=args.num_hidden, name='fc1')
    data = mx.symbol.SoftmaxOutput(data=data, name='softmax')

    return data


def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor, base_lr=factor) if len(step_) else None


def train_batch_end_callback(params):
    # print('end batch')
    
    # params.eval_metric.get_metric(0) # acc
    # params.eval_metric.get_metric(1) # ce
    
    # speedometer = mx.callback.Speedometer(args.batch_size, args.frequent)
    # logmetrics = mx.contrib.tensorboard.LogMetricsCallback(args.train_log_path)

    # if params.eval_metric.get_metric(0).global_num_inst % (args.batch_size * 10) == 0:
    #     logmetrics(params)
    # else:
    #     speedometer(params)
    
    acc = params.eval_metric.get_name_value()[0][1]
    ce = params.eval_metric.get_name_value()[1][1]

    global global_steps
    

    # 1 epoch 당 기록
    # if int(global_steps % (args.num_examples_eval/args.batch_size)) == 0:
    if int(global_steps % int(args.num_examples/args.batch_size)) == 0 and global_steps != 0:
        # with SummaryWriter(logdir=os.path.join(args.model_root, 'train_summary_' + str(args.number_of_layers))) as sw:
            #sw.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
        sw.add_scalar(tag='accuracy', value=acc, global_step=global_steps)     # params.nbatch
        sw.add_scalar(tag='cross-entropy', value=ce, global_step=global_steps)
        sw.add_scalar(tag='error', value=1-acc, global_step=global_steps)
        print('*** {0}/{1} epoch, {2} batch, acc: {3}, ce: {4}, err: {5}'.format(params.epoch, args.num_epoch, params.nbatch, acc, ce, 1-acc))
    # train 시에는 3번 나눠서 프린트 하도록
    if int(global_steps % int(args.num_examples/args.batch_size/3)) == 0:
        # speedometer(params)
        print('{0}/{1} epoch, {2} batch, acc: {3}, ce: {4}, err: {5}'.format(params.epoch, args.num_epoch, params.nbatch, acc, ce, 1-acc))

    global_steps = global_steps + 1


def eval_batch_end_callback(params):

    global best_ac
    global best_ce
    global best_epoch
    global best_batch

    global global_steps_eval

    acc = params.eval_metric.get_name_value()[0][1]
    ce = params.eval_metric.get_name_value()[1][1]
    err = 1 - acc

    # 1 epoch 당 기록
    if int(global_steps_eval % int(args.num_examples_eval/args.batch_size)) == 0 and global_steps_eval != 0:
        # with SummaryWriter(logdir=os.path.join(args.model_root, 'train_summary_' + str(args.number_of_layers))) as sw:
            #sw.add_pr_curve(tag='pseudo_pr_curve', predictions=predictions, labels=labels, num_thresholds=120)
        eval_sw.add_scalar(tag='accuracy', value=acc, global_step=global_steps - 1)
        eval_sw.add_scalar(tag='cross-entropy', value=ce, global_step=global_steps - 1)
        eval_sw.add_scalar(tag='error', value=err, global_step=global_steps - 1)
        print('*** eval {0}/{1} epoch, {2} batch, acc: {3}, ce: {4}, err: {5}'.format(params.epoch, args.num_epoch, params.nbatch, acc, ce, err))
    
    if int(global_steps_eval % int(args.num_examples_eval/args.batch_size)) == 0:
        print('--- eval {0}/{1} epoch, {2} batch, acc: {3}, ce: {4}, err: {5}'.format(params.epoch, args.num_epoch, params.nbatch, acc, ce, err))

    if best_ac < acc:
        best_ac = acc
        best_ce = ce
        best_epoch = params.epoch
        best_batch = params.nbatch
    
    global_steps_eval = global_steps_eval + 1


if __name__ == "__main__":

    ### config ###
    # data, 
    global args
    args = parse_args()
    set_config(args)
    print('config setting complete..')
    # mx_eval_metric = mx.metric.Accuracy()

    # summary writer
    global sw, sw_eval
    log_file_path = os.path.join(args.log_root, 'resnet_' + str(args.number_of_layers))
    log_file_path_eval = os.path.join(args.log_root, 'eval_resnet_' + str(args.number_of_layers))
    if not os.path.exists(log_file_path):
        os.mkdir(log_file_path)
    if not os.path.exists(log_file_path_eval):
        os.mkdir(log_file_path_eval)
    sw = SummaryWriter(logdir=log_file_path)
    eval_sw = SummaryWriter(logdir=log_file_path_eval)

    # log
    logger = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.ticker').disabled = True
    logger.propagate = True
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(args.log_root, 'resnet_' + str(args.number_of_layers) + '.log'), mode = "w")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # kv
    kvstore = mx.kvstore.create('device')
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    # batch_size
    batch_size = args.batch_size
    # epoch_size
    epoch_size = max(int(args.num_examples / args.batch_size / kvstore.num_workers), 1)
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0

    # checkpoint
    if not os.path.exists(os.path.join(args.model_root, "renset_" + str(args.number_of_layers))):
        os.mkdir(os.path.join(args.model_root, "renset_" + str(args.number_of_layers)))

    model_prefix = os.path.join(args.model_root, "renset_" + str(args.number_of_layers), "resnet-{}-{}-{}".format(args.data_set, args.number_of_layers, kvstore.rank))
    checkpoint = mx.callback.do_checkpoint(model_prefix, 1)
    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.model_load_epoch)

    # iter
    # cifar10_train data shape = 3,28,28
    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_train.rec") if args.data_set == 'cifar10' else
                              os.path.join(args.data_dir, "train_256_q90.rec") if args.aug_level == 1
                              else os.path.join(args.data_dir, "train_480_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 32, 32) if args.data_set == "cifar10" else (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 4 if args.data_set == "cifar10" else 0,  # 논문 cifar10 만 4pixel padded
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True, # 논문
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if args.data_set == "cifar10" else 1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if args.data_set == "cifar10" else 0 if args.aug_level == 1 else 0.25,
        random_h            = 0 if args.data_set == "cifar10" else 0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if args.data_set == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if args.data_set == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if args.aug_level <= 2 else 0.1,
        rand_mirror         = True, # flipped along the horizontal axis
        shuffle             = True,
        num_parts           = kvstore.num_workers,
        part_index          = kvstore.rank)

    # val data shape 3, 28, 28
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_val.rec") if args.data_set == 'cifar10' else
                              os.path.join(args.data_dir, "val_256_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 32, 32) if args.data_set=="cifar10" else (3, 224, 224),
        rand_crop           = False,    # val에서는 only use original 32x32 image
        rand_mirror         = False,
        num_parts           = kvstore.num_workers,
        part_index          = kvstore.rank)


    # symbol
    resnet_symbol = resnet()
    
    # feedforward
    # model = mx.model.FeedForward(
    #     ctx                 = ctx,
    #     symbol              = resnet_symbol,
    #     arg_params          = arg_params,
    #     aux_params          = aux_params,
    #     num_epoch           = 200 if args.data_set == "cifar10" else 120,
    #     begin_epoch         = begin_epoch,
    #     learning_rate       = args.lr,
    #     momentum            = args.momentum,
    #     wd                  = args.wd,
    #     # optimizer           = 'nag',
    #     optimizer          = 'sgd',
    #     initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
    #     lr_scheduler        = multi_factor_scheduler(begin_epoch, epoch_size, step=[120, 160], factor=0.1)
    #                          if args.data_set=='cifar10' else
    #                          multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90], factor=0.1),
    #     )

    # module
    #model = mx.model.Module()
    model = mx.mod.Module(symbol=resnet_symbol, context=ctx, data_names=['data'], label_names=['softmax_label'])

    #model.bind(data_shapes=[('data', (1, 3, 32, 32))], label_shapes=train.provide_label)
    model.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)

    if arg_params is not None and aux_params is not None:
        model.set_params(arg_params, aux_params)
    else:
        model.init_params(initializer=mx.init.Xavier(magnitude=2.), allow_missing=True, force_init=True) 
        # initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))

    model.init_optimizer(optimizer='sgd'
        , optimizer_params=(('learning_rate', 0.1)
            , ('lr_scheduler', multi_factor_scheduler(begin_epoch, epoch_size, step=[120, 160], factor=0.1)
                             if args.data_set=='cifar10' else
                             multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90], factor=0.1))
            , ('wd', args.wd), ('momentum', args.momentum)
            ))

    # fit
    model.fit(
        train,
        eval_data = val,
        # optimizer = 'sgd',
        # optimizer_params = {'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd},
        num_epoch = args.num_epoch, # 200 if args.data_set == "cifar10" else 120,
        begin_epoch = begin_epoch,
        # for report
        eval_metric        = ['acc', 'ce'] if args.data_set=='cifar10' else ['acc', mx.metric.create('top_k_accuracy', top_k = 5)],
        # eval_metric        = ['acc'] if args.data_set=='cifar10' else ['acc', mx.metric.create('top_k_accuracy', top_k = 5)],
        kvstore            = kvstore,
        # batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),   # print
        batch_end_callback = train_batch_end_callback,
        eval_batch_end_callback = eval_batch_end_callback,
        # batch_end_callback = end_batch,
        # batch_end_callback = mx.callback.Speedometer(1, 2),
        epoch_end_callback = checkpoint,
    )

    # sw close
    sw.close()

    print('=' * 60)
    print('original resnet {0} (data: {1}) end!'.format(args.number_of_layers, data_set))

    print('=' * 60)
    # print best acc, ce
    print('Best Validation Accuracy: {0}, Best Validation Crosss-Entropy: {1}, At Epoch: {2}, Batch: {3}'.format(best_ac, best_ce, best_epoch, best_batch))

    # digraph = mx.visualization.plot_network(resnet_symbol)
    # digraph.view()
    # digraph.render()
