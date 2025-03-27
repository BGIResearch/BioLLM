#!/usr/bin/env python3
# coding: utf-8
"""
@file: gears_gene_express_emb.py.py
@description: 
@author: Ping Qiu
@email: qiuping1@genomics.cn
@last modified by: Ping Qiu

change log:
    2024/04/10  create file.
"""

def test_scfoundation_norman():
    from biollm.task.perturbation.gears_task import GearsTask
    from biollm.evaluate.bm_metrices_gears import calculate_metrices_gears
    from biollm.repo.scfoundation.GEARS.gears import GEARS as scFoundationGEARS
    task = GearsTask('../../../biollm/config/pert/gears_scfoundation_exp_emb.toml')
    pert_data = task.make_pert_data()
    model_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/model/sc_foundation'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/model/sc_foundation/sc_foundation.pk'
    model = scFoundationGEARS(pert_data, device='cuda')
    model.model_initialize()
    model.load_pretrained(model_path)
    calculate_metrices_gears(pert_data, model, output_path)


def test_gears_norman():
    from biollm.evaluate.bm_metrices_gears import load_data, calculate_metrices_gears
    from biollm.repo.gears import GEARS
    data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/norman_gears'
    data_name = 'norman'
    model_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/model/gears'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/evaluate/norman/gears.pk'
    pert_data = load_data(data_path, data_name)
    model = GEARS(pert_data, device='cuda')
    model.model_initialize(pretrain_emb_type='universal')
    model.load_pretrained(model_path)
    calculate_metrices_gears(pert_data, model, output_path)


def test_gears_adamson():
    from biollm.evaluate.bm_metrices_gears import load_data, calculate_metrices_gears
    from biollm.repo.gears import GEARS
    data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson'
    data_name = 'adamson'
    model_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson/gears.pk'
    pert_data = load_data(data_path, data_name)
    model = GEARS(pert_data, device='cuda:2')
    model.model_initialize(pretrain_emb_type='universal')
    model.load_pretrained(model_path)
    test_res, test_metrics, test_pert_res = calculate_metrices_gears(pert_data, model, output_path)
    print(test_metrics)


def test_gears_mamba_adamson():
    from biollm.evaluate.bm_metrices_gears import load_data, calculate_metrices_gears
    from biollm.repo.gears import GEARS
    data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scmamba'
    data_name = 'adamson'
    model_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scmamba'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scmamba/gears.pk'
    pert_data = load_data(data_path, data_name)
    model = GEARS(pert_data, device='cuda:2')
    model.model_initialize(pretrain_emb_type='universal')
    model.load_pretrained(model_path)
    test_res, test_metrics, test_pert_res = calculate_metrices_gears(pert_data, model, output_path)
    print(test_metrics)

def test_gears_scgpt_adamson():
    from biollm.evaluate.bm_metrices_gears import load_data, calculate_metrices_gears
    from biollm.repo.gears import GEARS
    data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scmamba'
    data_name = 'adamson'
    model_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scgpt'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scgpt/gears.pk'
    pert_data = load_data(data_path, data_name)
    model = GEARS(pert_data, device='cuda:2')
    model.model_initialize(pretrain_emb_type='universal')
    model.load_pretrained(model_path)
    test_res, test_metrics, test_pert_res = calculate_metrices_gears(pert_data, model, output_path)
    print(test_metrics)


def test_gears_scfoundation_adamson():
    from biollm.evaluate.bm_metrices_gears import load_data, calculate_metrices_gears
    from biollm.repo.gears import GEARS
    data_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scmamba'
    data_name = 'adamson'
    model_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scfoundation'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/data/finetune/gears_data/adamson_scfoundation/gears.pk'
    pert_data = load_data(data_path, data_name)
    model = GEARS(pert_data, device='cuda:2')
    model.model_initialize(pretrain_emb_type='universal')
    model.load_pretrained(model_path)
    test_res, test_metrics, test_pert_res = calculate_metrices_gears(pert_data, model, output_path)
    print(test_metrics)


def test_scmamba_norman():
    from biollm.task.perturbation.gears_task import GearsTask
    from biollm.evaluate.bm_metrices_gears import calculate_metrices_gears
    from biollm.repo.gears import GEARS
    task = GearsTask('../../../biollm/config/pert/gears_mamba_gene-express.toml')
    pert_data = task.make_pert_data()
    gene_emb_weight = task.universal_gene_embdedding(pert_data)
    model_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/biollm/result/pert/mamba/gears_test'
    output_path = '/home/share/huadjyin/home/s_qiuping1/workspace/omics_model/bio_model/biollm/biollm/result/pert/mamba/gears_test/sc_mamba.pk'
    gears_model = GEARS(pert_data, device=task.device, model_output=task.args.result_dir)
    gears_model.model_initialize(hidden_size=task.args.hidden_size, use_pretrained=task.args.use_pretrained,
                                 pretrain_freeze=task.args.pretrain_freeze, gene_emb_weight=gene_emb_weight,
                                 pretrained_emb_size=task.args.pretrained_emb_size, model_loader=task.load_obj,
                                 pretrain_emb_type=task.args.emb_type)
    gears_model.load_pretrained(model_path)
    calculate_metrices_gears(pert_data, gears_model, output_path)


if __name__ == '__main__':
    test_gears_scgpt_adamson()
