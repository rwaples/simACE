rule prepare_weibull:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.weibull.parquet"
    output:
        data="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/data.dat",
        codelist="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/codelist.txt",
        varlist="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/varlist.txt",
        pedigree_ped="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/pedigree.ped",
        weibull_config="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/weibull.txt"
    log:
        "logs/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/prepare.log"
    script:
        "../scripts/prepare_weibull.py"


rule run_weibull:
    input:
        data="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/data.dat",
        codelist="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/codelist.txt",
        varlist="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/varlist.txt",
        pedigree_ped="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/pedigree.ped",
        weibull_config="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/weibull.txt"
    output:
        rwe="results/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/results.rwe"
    log:
        "logs/analysis/{folder}/{scenario}/rep{rep}/weibull/trait{trait}/run.log"
    shell:
        "cd results/analysis/{wildcards.folder}/{wildcards.scenario}/rep{wildcards.rep}/weibull/trait{wildcards.trait} "
        "&& /home/ryanw/Documents/ACE/external/Survival_Kit_executables_Linux/weibull.exe "
        "> ../../../../../../{log} 2>&1"
