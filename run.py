import sys

ROOT_DIR = '.'
#ROOT_DIR = '/home/ubuntu/ImgGen/task2'

sys.path.insert(0, f"{ROOT_DIR}/src")

from src.neural_nets import *
from src.utils import *
from src.eval_utils import *
from src.train_utils import *
from time import time

import json
from torch.utils.data import DataLoader

CONFIG_FILE_JSON = f'{ROOT_DIR}/src/learning_config.json'
LOGS_DIR = f'{ROOT_DIR}/logs'
DATA_DIR = f'{ROOT_DIR}/data'
IMAGES_DIR = f'{DATA_DIR}/LFW-yolo'
IMAGES_INFO = f'{DATA_DIR}/images_info.csv'
PAIRS_TABLE = f'{DATA_DIR}/pairs.csv'

##############################

with open(CONFIG_FILE_JSON, 'r', encoding='utf-8') as fd:
    json_obj = json.loads(fd.read())
learn_config = LearningConfig(**json_obj)

##############################

SAVE_DIR = f"{LOGS_DIR}/{learn_config.run_name}"
PLOTS_DIR = f'{SAVE_DIR}/plots'
BEST_MODEL_SAVE_PATH = f"{SAVE_DIR}/best_model.pt"
LAST_MODEL_SAVE_PATH = f"{SAVE_DIR}/last_model.pt"
LOGS_PATH = f"{SAVE_DIR}/logs.txt"
os.mkdir(SAVE_DIR)
os.makedirs(PLOTS_DIR)

print("Saving used config")
with open(f"{SAVE_DIR}/used_config.json", 'w', encoding='utf-8') as fd:
    json.dump(learn_config.__dict__, indent=2, fp=fd)

##############################

print("Init train objectives")

num_classes = len(os.listdir(IMAGES_DIR))
print(num_classes)

IMAGE_PREPROCESSOR = AutoImageProcessor.from_pretrained(VIT_PATH if learn_config.backbone == 'vit' else EFFICIENTNET_PATH)

print("Saving used preprocessor")
with open(f"{SAVE_DIR}/used_prep.txt", 'w', encoding='utf-8') as fd:
    fd.write("\n=======EMBEDDER_ARCH=======\n")
    fd.write(IMAGE_PREPROCESSOR.__str__())

embed_model = EmbedderNet(EMBED_SIZE, learn_config.backbone).to(learn_config.device)
cls_model = MetricNet(learn_config.head_name, EMBED_SIZE, num_classes, learn_config.arcface_s, 
                             learn_config.archface_m, device=learn_config.device)

print("Saving used nn-arch")
with open(f"{SAVE_DIR}/used_arch.txt", 'w', encoding='utf-8') as fd:
    fd.write("\n=======EMBEDDER_ARCH=======\n")
    fd.write(embed_model.__str__())
    fd.write("\n=======METRIC_ARCH=======\n")
    fd.write(cls_model.__str__())

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW([{'params': embed_model.parameters()}, {'params': cls_model.parameters()}],
                            lr=learn_config.lr, weight_decay=learn_config.weight_decay)

if learn_config.use_scheduler == 'yes':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learn_config.lr_step, 
                                                gamma=learn_config.scheduler_gamma)
else:
    scheduler = None

print("Parameters count: ",param_count(embed_model))

##############################

print("Load train/eval datasets")

train_dataset = CustomPeopleDataset(IMAGES_INFO, IMAGE_PREPROCESSOR, base_dir=ROOT_DIR)
train_loader = DataLoader(train_dataset, batch_size=learn_config.batch, 
                          collate_fn=custom_collate, shuffle=True)

eval_dataset = CSVBenchmarkDataset(root=IMAGES_DIR,
                              csv_file=PAIRS_TABLE,
                              transform=IMAGE_PREPROCESSOR,
                              return_img_paths=False)

##############################

ml_train = []
eval_scores = []
best_score = 0

for i in range(learn_config.epochs):
    print(f"Epoch {i+1} start:")

    train_s = time()
    train_losses = train(embed_model, cls_model, train_loader, optimizer, scheduler, 
                         criterion, learn_config.device)
    train_e = time()

    eval_info = run_benchmarking(eval_dataset, embed_model, learn_config.device, 
                                         PLOTS_DIR, i)
    eval_e = time()

    if eval_info['cosine'][0] > best_score:
        best_score = eval_info['cosine'][0]
        torch.save(embed_model.state_dict(), BEST_MODEL_SAVE_PATH)

    #
    ml_train.append(np.mean(train_losses))
    eval_scores.append(eval_info)
    print(f"Epoch {i+1} results: tain_loss - {round(ml_train[-1], 5)}")
    print(eval_scores[-1])

    # Save train/eval info to logs folder
    epoch_log = {
        'epoch': i+1, 'train_loss': ml_train[-1],
        'cosine': eval_scores[-1]['cosine'], 'euclidian': eval_scores[-1]['euclidian'],
        'train_time': round(train_e - train_s, 5), 'eval_time': round(eval_e - train_e, 5)
        }
    with open(LOGS_PATH,'a',encoding='utf-8') as logfd:
        logfd.write(str(epoch_log) + '\n')

torch.save(embed_model.state_dict(), LAST_MODEL_SAVE_PATH)