import hydra
from hydra import utils
from omegaconf import DictConfig

from train_planetoid import run as train_planetoid
from train_arxiv import run as train_arxiv
from train_ppi import run as train_ppi
from train_ppi_induct import run as train_ppi_induct


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg = cfg[cfg.key]
    root = utils.get_original_cwd()
    print(cfg)

    if cfg.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        train_planetoid(cfg, root)
    elif cfg.dataset == 'Arxiv':
        train_arxiv(cfg, root)
    elif cfg.dataset == 'PPI':
        train_ppi(cfg, root)
    elif cfg.dataset == 'PPIinduct':
        train_ppi_induct(cfg, root)


if __name__ == "__main__":
    main()