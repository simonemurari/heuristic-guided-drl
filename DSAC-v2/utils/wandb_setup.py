import wandb

def wandb_init(project, entity, group, name, config):
    wandb.init(
        project=project,
        entity=entity,
        sync_tensorboard=True,
        name=name,
        config=config,
        group=group
    )

def add_scalar(tb_info):
    wandb.log(tb_info)   

def wandb_finish():
    wandb.finish()