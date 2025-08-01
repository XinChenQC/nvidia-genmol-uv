import argparse
import json
import sys
from genmol.sampler import Sampler


def get_default_params(task):
    if task == 'random':
        return {'gamma': 0, 'randomness': 0.5, 'softmax_temp': 0.5}
    elif task == 'linker_design':
        return {'gamma': 0, 'randomness': 3, 'softmax_temp': 1.2}
    elif task == 'motif_extension':
        return {'gamma': 0.3, 'randomness': 1.2, 'softmax_temp': 1.2}
    elif task == 'scaffold_decoration':
        return {'gamma': 0.3, 'randomness': 2, 'softmax_temp': 1.2}
    elif task == 'superstructure_generation':
        return {'gamma': 0.4, 'randomness': 2, 'softmax_temp': 1.2}
    else:
        return {'gamma': None, 'randomness': None, 'softmax_temp': 1.2}

def run_task(args):
    import torch
    torch.set_num_threads(int(args.ncores))
    args.task = args.task.lower()
    defaults = get_default_params(args.task)
    gamma = args.gamma if args.gamma is not None else defaults['gamma']
    randomness = args.randomness if args.randomness is not None else defaults['randomness']
    softmax_temp = args.softmax_temp if args.softmax_temp is not None else defaults['softmax_temp']

    sampler = Sampler(args.model_path)

    if args.task in ['scaffold_decoration', 'motif_extension', 'superstructure_generation']:
        samples = sampler.fragment_completion(
            args.fragment,
            num_samples=args.num_samples,
            gamma=gamma,
            randomness=randomness,
            softmax_temp=softmax_temp
        )
    elif args.task == 'random':
        samples = sampler.de_novo_generation(
            num_samples=args.num_samples,
            randomness=randomness,
            softmax_temp=softmax_temp
        )
    elif args.task == 'linker_design':
        samples = sampler.fragment_linking(
            args.fragment,
            num_samples=args.num_samples,
            randomness=randomness
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return list(set(samples))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--fragment', type=str, required=True)
    parser.add_argument('--ncores', type=str, default=12)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--model-path', type=str, default='./model.ckpt')
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--randomness', type=float, default=None)
    parser.add_argument('--softmax-temp', type=float, default=None)
    args = parser.parse_args()

    samples = run_task(args)
    print(json.dumps(samples))

if __name__ == "__main__":
    main()
