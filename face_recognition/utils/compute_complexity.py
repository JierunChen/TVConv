import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir, os.pardir))
sys.path.insert(0, parent_dir_path)
from face_recognition.model import *
import torch
from ptflops import get_model_complexity_info
import os
from face_recognition.model_interface import LitModel
from argparse import ArgumentParser
import torch.autograd.profiler as profiler


def flops_to_string(flops, units='MMac', precision=2):
    if units is None:
        if flops // 10 ** 9 > 0:
            return str(round(flops / 10. ** 9, precision)) + ' GMac'
        elif flops // 10 ** 6 > 0:
            return str(round(flops / 10. ** 6, precision)) + ' MMac'
        elif flops // 10 ** 3 > 0:
            return str(round(flops / 10. ** 3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10. ** 9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10. ** 6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10. ** 3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units='K', precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def running_time(use_gpu, model, input_size, device, batch_size, runs):
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        # input = torch.autograd.Variable(torch.rand(batch_size, *input_size), device=device)
        input = torch.rand((batch_size, *input_size), device=device)
        model = model.to(device)
        sort_name = "cuda_time_total"
    else:
        # input = torch.autograd.Variable(torch.rand(batch_size, *input_size))
        input = torch.rand(batch_size, *input_size)
        sort_name = "cpu_time_total"
    # script_cell = torch.jit.trace(model, [input])
    # script_cell(input)
    for _ in range(runs):
        model(input)
    # with profiler.profile(use_cuda=use_gpu, profile_memory=True) as prof:
    with profiler.profile(use_cuda=use_gpu, profile_memory=False) as prof:
        with profiler.record_function("conv_"+str(runs)+'_runs'):
            for _ in range(runs):
                model(input)
                # script_cell(input)
    print(prof.key_averages().table(sort_by=sort_name, row_limit=10))


def complexity(args):
    if args.threads>0:
        torch.set_num_threads(args.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model_transform = False

    checkpoint_path = args.model_ckpt_dir + args.load_from_ckpt_path
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        hp = checkpoint['hyper_parameters']
        model = LitModel(10572, hp)
        model.load_state_dict(checkpoint['state_dict'])
        if hp.atom=='TVConv':
            model_transform = True
    else:
        lit_model = LitModel(10572, args)
        if args.atom=='TVConv':
            model_transform = True
    if model_transform:
    # if False:
        print('model transformed!')
        model = model_transform_for_test(lit_model.model)
    else:
        model = lit_model.model
    model.eval()
    with torch.no_grad():
        # print(model)

        input_size = (3, 96, 96)
        running_time(use_gpu=bool(args.use_gpu),
                     model=model,
                     input_size=input_size,
                     device=torch.device("cuda:0"),
                     batch_size=1,
                     runs=100
        )
        """==================================================="""
        """==================================================="""
        """==================================================="""

        # cus_mapping = {TVConv_test: TVConv_test_flops_counter_hook,
        #                CondConv2d: CondConv2d_flops_counter_hook,
        #                nn.Sigmoid: sigmoid_flops_counter_hook,
        #                conv2d_sample_by_sample_class: conv2d_sample_by_sample_class_flops_counter_hook,
        #                ele_multiply: ele_multiply_flops_counter_hook,
        #                torch_mm: torch_mm_flops_counter_hook,
        #                nn.Softmax: softmax_flops_counter_hook,
        #                dynamic_conv2d_sample: dynamic_conv2d_sample_flops_counter_hook,
        #                conv2d_sample: conv2d_sample_flops_counter_hook,
        #                unfold_conv: unfold_conv_flops_counter_hook,
        #     }
        # macs, params = get_model_complexity_info(model, input_size, as_strings=False,
        #                                          print_per_layer_stat=False, verbose=False,
        #                                          # print_per_layer_stat=True, verbose=True,
        #                                          custom_modules_hooks=cus_mapping)
        # macs = flops_to_string(macs)
        # params = params_to_string(params)
        # print('{:<30}  {:<12}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<12}'.format('Number of parameters: ', params))

        """==================================================="""
        """==================================================="""
        """==================================================="""


if __name__ == "__main__":
    parser = ArgumentParser()

    # NORMAL args
    parser.add_argument("--input_size", type=list, default=[96, 96])
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument("--load_from_ckpt_path", type=str, default='')
    parser.add_argument("--gpus", type=int, default=-1,
                        help='how many gpu used among the visible gpus')

    # frequently reset args
    parser.add_argument('-m', "--model_name", type=str, default="mobilenet_v2_x1_0")
    parser.add_argument('-g', "--gpu_id", type=str, default='0', help='visible gpu id')
    parser.add_argument('-a', "--atom", type=str, default='base')
    # parser.add_argument('-a', "--atom", type=str, default='TVConv', choices=['TVConv', 'base', 'invo', 'CondConv'])
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--use_gpu", type=int, default=0)

    # TVConv related
    parser.add_argument("--TVConv_k", type=int, default=3)
    parser.add_argument("--TVConv_gk", type=int, default=3, help='kernel size of the generating block')
    parser.add_argument("--TVConv_posi_chans", type=int, default=4, help='affinity maps')
    parser.add_argument("--TVConv_inter_chans", type=int, default=64)
    parser.add_argument("--TVConv_inter_layers", type=int, default=3)
    parser.add_argument("--TVConv_Bias", type=int, default=0, choices=[0, 1])
    parser.add_argument("--TVConv_BN", type=str, default='LN', choices=['None', 'BN', 'LN', 'IN'])
    parser.add_argument("--TVConv_posi0wd", type=int, default=0, choices=[0, 1],
                        help='no weight decay for affinity maps')
    parser.add_argument("--TVConv_FN", type=int, default=0, choices=[0, 1])

    parser.add_argument("--invo_group_channels", type=int, default=8)

    args = parser.parse_args()

    model_list=[
        'mobilenet_v2_x0_1',
        'mobilenet_v2_x0_1',
        'mobilenet_v2_x0_2',
        'mobilenet_v2_x0_3',
        'mobilenet_v2_x0_5',
        'mobilenet_v2_x1_0'
        # 'shufflenet_v2_x0_5',
        # 'shufflenet_v2_x1_0',
        # 'shufflenet_v2_x1_5',
    ]

    print('------------------------------------------------------------', args.atom)
    for model in model_list:
        print(model)
        args.model_name = model
        complexity(args)