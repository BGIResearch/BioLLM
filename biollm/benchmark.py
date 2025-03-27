#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: benchmark.py
@time: 2025/3/5 17:12
"""
import time
import pickle
import toml
from tqdm import tqdm
from functools import wraps
from biollm.tasks.cell_embedding import CellEmbTask
from biollm.tasks.annotation.universal_anno import AnnotationTask


def dumps_result(output, res):
    with open(output, 'wb') as fd:
        pickle.dump(res, fd)


def time_logger(func):
    """
    计时装饰器，统计任务执行时间，并打印日志
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        task_name = kwargs.get("task_name", "Task")
        print(f"Starting {task_name}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - start_time
        print(f"{task_name} completed in {total_time:.2f} seconds.\n")
        return result

    return wrapper


@time_logger
def run_task(config_dict, TaskClass, task_name):
    """
    通用任务执行函数
    """
    result = {}

    for model in tqdm(config_dict, desc=f"Processing {task_name}"):
        print(f" Running {model}...")
        if model == "datasets":  # 跳过数据集
            continue
        dataset_res = {}
        datasets = config_dict["datasets"]
        for i in range(len(config_dict[model])):
            conf = config_dict[model][i]
            task_start = time.time()
            try:
                task_obj = TaskClass(config_file=conf)
                metrics = task_obj.run()
                dataset_res[datasets[i]] = metrics
            except Exception as e:
                print(f'Error: {datasets[i]}', e)

            task_time = time.time() - task_start
            print(f"  {task_name} | {model} | {config_dict['datasets'][i]} | Time: {task_time:.2f}s")

        result[model] = dataset_res

    return result


def main(config_file):
    with open(config_file, 'r') as f:
        config = toml.load(f)
    print('Configs: ', config)
    output_dir = config['output_dir']

    task_map = {
        "Annotation": (run_task, AnnotationTask, "annotation_bench.pk"),
        "GRN": (run_task, None, "grn_bench.pk"),
        "Imputation": (run_task, None, "imputation_bench.pk"),
        "Perturbation": (run_task, None, "perturbation_bench.pk"),
        "CellEmbedding": (run_task, CellEmbTask, "cell_emb_bench.pk"),
    }

    for task_name, (runner, TaskClass, output_file) in task_map.items():
        if task_name in config and TaskClass:
            task_res = runner(config[task_name], TaskClass, task_name=task_name)
            dumps_result(f"{output_dir}/{output_file}", task_res)


def summary_cell_emb():
    pass


if __name__ == '__main__':
    path = '/home/share/huadjyin/home/s_qiuping1/workspace/BioLLM2/biollm/config/benchmark.toml'
    main(path)
