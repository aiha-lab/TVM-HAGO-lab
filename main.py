# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from collections import namedtuple
import tvm
from tvm import relay, autotvm
from tvm import hago
import mxnet as mx
import numpy as np
from mxnet import gluon
import logging
import os
import pickle
import time
import argparse
import tvm.relay.testing
from tvm.contrib.debugger import debug_runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner


logging.basicConfig(level=logging.DEBUG)

Config = namedtuple('Config', ['model', 'expected_acc'])
# dev = tvm.device(target)

# argparse
parser = argparse.ArgumentParser(description = 'argparse')

parser.add_argument('--dtype', '-p',  required=False, default='fp32', help='fp32 or int8')
parser.add_argument('--tune', '-t', required=False, default=False, help='tune or not')
parser.add_argument('--debug', '-d',  required=False, default=False, help='use debugger')

args = parser.parse_args()

#### TUNING OPTION ####
network = "resnet-18"
log_file = "%s-%s.log" % (network, args.dtype)
dtype = "float32"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 2000,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def get_val_data(model_name,
                 rec_val,
                 batch_size,
                 num_workers=4):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 299 if model_name == 'inceptionv3' else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn


def get_model(model_name, batch_size, qconfig, original=False, simulated=False, dataset=None):
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})

    graph = hago.prerequisite_optimize(mod['main'], params=params)
    logging.debug('original')
    logging.debug(graph.astext(show_meta_data=False))

    if original:
        return graph, params

    with qconfig:
        logging.debug('current quantize config')
        logging.debug(hago.current_qconfig())
        # import pdb; pdb.set_trace()
        hardware = hago.create_accelerator_description()
        space = hago.generate_search_space(graph, hardware)
        # tuner = hago.BatchedGreedySearchTuner(space, 'accuracy')
        tuner = hago.DefaultSetting(space, 'accuracy')
        # tuner = hago.GreedySearchTuner(space, 'accuracy') 
        ctx = tvm.gpu()
        target = 'cuda'
        strategy, result = hago.search_quantize_strategy(graph, hardware, dataset, tuner, ctx, target)
      
        quantizer = hago.create_quantizer(graph, hardware, strategy)
        simulated_graph = quantizer.simulate()
        quantized_graph = quantizer.quantize()
        canonic_graph = relay.qnn.transform.CanonicalizeOps()(tvm.IRModule.from_expr(quantized_graph))
        logging.debug('simulated graph')
        logging.debug(simulated_graph.astext(show_meta_data=False))
        logging.debug('quantize graph')
        logging.debug(quantized_graph.astext(show_meta_data=False))
        # logging.debug('canonicalized graph')
        # logging.debug(canonic_graph.astext(show_meta_data=False))
        # hago.inspect_graph_statistic(graph, hardware, strategy, dataset, ctx, target)
        return quantized_graph, params


def tune_eval(mod, params, dataset, batch_fn, tuning_opt, target='cuda', ctx=tvm.gpu(), log_interval=100):

    # extract workloads from relay program
    print("Extract tasks...")
    
    # mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    # mod = relay.Function(
    #     mod.params, relay.nn.softmax(mod.body), None, mod.type_params, mod.attrs
    # )
    
    mod = tvm.IRModule.from_expr(mod)
    batch_size = 32
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    
    tune_start = time.time()
    if args.tune:    
        tasks = autotvm.task.extract_from_program(
            mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
        )
        
        # run tuning tasks
        print("Tuning...")
        # import pdb; pdb.set_trace()
        tune_tasks(tasks, **tuning_opt)
        
        '''
        # compile kernels with history best records
        with autotvm.apply_history_best(log_file):
            print("Compile...")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod, target=target, params=params)

            # load parameters
            dev = tvm.device(str(target), 0)
            module = runtime.GraphModule(lib["default"](dev))
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            module.set_input("data", data_tvm)
        '''
    tune_end = time.time()
    print("Tuning time : %.3f"%(tune_end-tune_start))

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target)
        
        if not args.debug:
            # create runtime module
            m = tvm.contrib.graph_runtime.create(graph, lib, ctx)

        elif args.debug:
            # create debug runtime
            m = debug_runtime.create(graph, lib, ctx, dump_root="/tmp/tvmdbg")
        
        m.set_input(**params)

        # executor = relay.create_executor("vm", tvm.IRModule.from_expr(mod), ctx ,target)
 
        ######################################################################
        # Time evaluator
        e = m.module.time_evaluator("run", ctx, repeat=3)
        t = np.array(e().results)*1000
        print("time_evaluator: %.3fms (%.5fms)"%(t.mean(), t.std()))
        ######################################################################

        '''
        # setup evaluaiton metric
        dataset.reset()
        batch_size = dataset.batch_size
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        acc_top1.reset()
        acc_top5.reset()

        t_arr = []
        # Execute
        start = time.time()
        for i, batch in enumerate(dataset):
            # print("batch id : ", i)
            # t0 = time.time()
            
            data, label = batch_fn(batch, [mx.cpu(0)])
            # data = data[0].asnumpy()
            t1 = time.time()
            # print("batch_fn : %.3f"%(t1-t0))
            
            # import pdb; pdb.set_trace()
            # out_arr = executor.evaluate()(data)
            m.run(data=data[0].asnumpy())
            # t2 = time.time() 
            # print("m.run : %.3f"%(t2-t1))

            out_arr = m.get_output(0)
            # t3 = time.time()
            # print("get_output : %.3f"%(t3-t2))
            t_elapsed = time.time() - t1
            t_arr.append(t_elapsed)
            
            acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
            acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])
            # t4 = time.time()
            # print("accuracy update : %.3f"%(t4-t3))
            
            if not (i + 1) % log_interval:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                nsamples = (i + 1) * batch_size
                logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)
            # print("batch %d time : %.3fs" % (i, time.time()-t0))
        
        end = time.time()
        '''
    
    # logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    
    ########################################################
    # t_arr = np.array(t_arr[1:])*1000
    # print("size:",t_arr.size)
    # print(t_arr)
    # print("latency : %.3fms(per image) (%d-batch: %.3fms(%.3fms))"%(t_arr.mean()/batch_size, batch_size, t_arr.mean(), t_arr.std()))
    # print("Evaluation time : %.3fs"%(end-start))
    ########################################################
    # print("Tuning time : %.3f"%(tune_end-tune_start))
    # return top1


def get_calibration_dataset(dataset, batch_fn, num_samples=100):
    dataset.reset()
    batches = []
    for i, batch in enumerate(dataset):
        if i * dataset.batch_size > num_samples:
            break
        data, label = batch_fn(batch, [mx.cpu(0)])
        batches.append({'data': tvm.nd.array(data[0].asnumpy()),
                        'label': tvm.nd.array(label[0].asnumpy())})
    return hago.CalibrationDataset(batches)


def test_quantize_acc(cfg, rec_val):
    # qconfig = hago.qconfig(skip_conv_layers=[0], log_file='temp.log')
    qconfig = hago.qconfig(use_channel_quantize=False, log_file='temp.log')
    batch_size = 32
    val_data, batch_fn = get_val_data(cfg.model, rec_val=rec_val, batch_size=batch_size)
    dataset = get_calibration_dataset(val_data, batch_fn)
    
    orig = True
    
    if(args.dtype == 'int8'):
        orig = False
        
    mod, params = get_model(cfg.model, batch_size, qconfig, dataset=dataset, original=orig)
    tune_eval(mod, params, val_data, batch_fn, tuning_option, target='cuda', ctx=tvm.gpu(0))
    
    # print("Final accuracy", "fp32" if orig else "int8", acc)

    # return acc


if __name__ == "__main__":
    #TODO(for user): replace the line with the path to imagenet validation dataset
    rec_val = "./val.rec"
    # rec_val = "~/tensorflow_datasets/downloads/manual/imagenet2012/val_rec.rec"

    results = []
    configs = [
        Config('resnet18_v1', expected_acc=0.69),
        # Config('resnet50_v1', expected_acc=0.75),
        # Config('inceptionv3', expected_acc=0.76),
        # Config('mobilenet1.0', expected_acc=0.70)
    ]
    # rec = hago.pick_best(".quantize_strategy_search.log", 'quant_acc')

    for config in configs:
        acc = test_quantize_acc(config, rec_val)
        results.append((config, acc))
    # for res in results:
        # print("{}\nQuantized Accuracy: {} vs. Expected Accuracy: {}".format(res[0].model, res[1], res[0].expected_acc))
