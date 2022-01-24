import hydra
from hydra import utils
from omegaconf import DictConfig
import mlflow
import torch

from train_planetoid import run as train_planetoid
from train_webkb import run as train_webkb
from train_arxiv import run as train_arxiv
from train_ppi import run as train_ppi
from train_ppi_induct import run as train_ppi_induct
from train_reddit import run as train_reddit


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        print('{}: {}'.format(param_name, element))
        mlflow.log_param(param_name, element)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg_mlflow = cfg.mlflow
    cfg = cfg[cfg.key]
    root = utils.get_original_cwd()

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mlflow.set_tracking_uri('http://' + cfg_mlflow.server_ip + ':5000')
    mlflow.set_experiment(cfg_mlflow.runname)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(cfg)
        if cfg.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            test_acces = train_planetoid(cfg, root, device)
        elif cfg.dataset in ['Cornell', 'Texas', 'Wisconsin']:
            test_acces = train_webkb(cfg, root, device)
        elif cfg.dataset == 'Arxiv':
            test_acces = train_arxiv(cfg, root, device)
        elif cfg.dataset == 'PPI':
            test_acces = train_ppi(cfg, root, device)
        elif cfg.dataset == 'PPIinduct':
            test_acces = train_ppi_induct(cfg, root, device)
        elif cfg.dataset == 'Reddit':
            test_acces = train_reddit(cfg, root, device)
        
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