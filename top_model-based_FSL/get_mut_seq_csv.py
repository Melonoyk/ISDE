from evolvepro.src.process import generate_wt, generate_single_aa_mutants_csv

dataset_ls = ['cas12f', 'brenan', 'cov2_S', 'doud', 'giacomelli', 'haddox', 'jones', 'kelsic', 'lee', 'markin', 'stiffler', 'zikv_E']
for dataset in dataset_ls:
    generate_single_aa_mutants_csv(
        f'/data1/users/weig03/data/Focus_work/EvolvePro-main/data/dms/wt_fasta/{dataset}_WT.fasta',
        f'/data1/users/weig03/data/Focus_work/EvolvePro-main/output/dms/{dataset}.csv'
    )
