import os
import sys


def create_logdir(hparams):
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = os.path.join(base_dir, f'result_{hparams.ratio}_{hparams.seed}')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    result_dirs = os.path.join(base_dir, hparams.dataset)
    os.makedirs(result_dirs, exist_ok=True)

    ulb_ratio_str = f"{hparams.ulb_ratio:.2f}"
    ulb_imb_type_str = hparams.ulb_imb_type
    imb_ratio_str = f"{hparams.imb_ratio:.1f}"
    filename = (
        ulb_ratio_str
        + ulb_imb_type_str
        + imb_ratio_str
        + '_doc_model_8_lr_'
        + f"{hparams.lr}"
        + '.csv'
    )
    return os.path.join(result_dirs, filename)
