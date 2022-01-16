import hydra
from hydra import utils
from omegaconf import DictConfig
import mlflow

from train_planetoid import run as train_planetoid
from train_arxiv import run as train_arxiv
from train_ppi import run as train_ppi
from train_ppi_induct import run as train_ppi_induct


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        mlflow.log_param(param_name, element)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg_mlflow = cfg.mlflow
    cfg = cfg[cfg.key]
    root = utils.get_original_cwd()
    print(cfg)

    mlflow.set_tracking_uri('http://' + cfg_mlflow.server_ip + ':5000')
    mlflow.set_experiment(cfg_mlflow.runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        if cfg.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            test_acces = train_planetoid(cfg, root)
        elif cfg.dataset == 'Arxiv':
            test_acces = train_arxiv(cfg, root)
        elif cfg.dataset == 'PPI':
            test_acces = train_ppi(cfg, root)
        elif cfg.dataset == 'PPIinduct':
            test_acces = train_ppi_induct(cfg, root)
        
        for i, acc in enumerate(test_acces):
            mlflow.log_metric('acc', value=acc, step=i)
        test_acc_mean = sum(test_acces)/len(test_acces)
        mlflow.log_metric('acc_mean', value=test_acc_mean)
        mlflow.log_metric('acc_max', value=max(test_acces))
        mlflow.log_metric('acc_min', value=min(test_acces))

    print('test mean acc: {:.3f}'.format(test_acc_mean))
    return test_acc_mean


if __name__ == "__main__":
    main()